import {
  AutoModel,
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  Tensor,
  TextStreamer,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

import { KokoroTTS, TextSplitterStream } from "https://cdn.jsdelivr.net/npm/kokoro-js@1.1.2/+esm";

import {
  EXIT_THRESHOLD,
  INPUT_SAMPLE_RATE,
  MAX_BUFFER_DURATION,
  MAX_NUM_PREV_BUFFERS,
  MIN_SILENCE_DURATION_SAMPLES,
  MIN_SPEECH_DURATION_SAMPLES,
  SPEECH_PAD_SAMPLES,
  SPEECH_THRESHOLD,
} from "./voice-constants.js";

function postStatus(status, message, extra = {}) {
  self.postMessage({ type: "voice_status", status, message, ...extra });
}

postStatus("loading", "Preparing voice pipeline...");

const device = "webgpu";
let voice = "af_heart";
let tts;
let silero_vad;
let transcriber;
let tokenizer;
let llm;
let messages;
let past_key_values_cache;
let stopping_criteria;

const SYSTEM_MESSAGE = {
  role: "system",
  content:
    "You are a friendly conversational assistant. Keep replies short, warm, and human. If user gives a robot command, still respond naturally.",
};

async function initModels() {
  postStatus("loading", "Downloading TTS model...");
  tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
    dtype: "fp32",
    device,
  });

  postStatus("loading", "Downloading VAD model...");
  silero_vad = await AutoModel.from_pretrained("onnx-community/silero-vad", {
    config: { model_type: "custom" },
    dtype: "fp32",
  });

  postStatus("loading", "Downloading ASR model...");
  transcriber = await pipeline("automatic-speech-recognition", "onnx-community/whisper-base", {
    device,
    dtype: {
      encoder_model: "fp32",
      decoder_model_merged: "fp32",
    },
  });

  postStatus("loading", "Warming up ASR...");
  await transcriber(new Float32Array(INPUT_SAMPLE_RATE));

  const llmModelId = "HuggingFaceTB/SmolLM2-1.7B-Instruct";
  postStatus("loading", "Downloading conversational model...");
  tokenizer = await AutoTokenizer.from_pretrained(llmModelId);
  llm = await AutoModelForCausalLM.from_pretrained(llmModelId, {
    dtype: "q4f16",
    device,
  });

  postStatus("loading", "Warming up conversational model...");
  await llm.generate({ ...tokenizer("x"), max_new_tokens: 1 });

  messages = [SYSTEM_MESSAGE];
  postStatus("ready", "Voice ready", { voices: tts.voices || {} });
}

const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;
const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
let isRecording = false;
let isPlaying = false;
let postSpeechSamples = 0;
let prevBuffers = [];

async function vad(buffer) {
  const input = new Tensor("float32", buffer, [1, buffer.length]);
  const { stateN, output } = await silero_vad({ input, sr, state });
  state = stateN;
  const isSpeech = output.data[0];
  return isSpeech > SPEECH_THRESHOLD || (isRecording && isSpeech >= EXIT_THRESHOLD);
}

function resetAfterRecording(offset = 0) {
  postStatus("recording_end", "Transcribing...");
  BUFFER.fill(0, offset);
  bufferPointer = offset;
  isRecording = false;
  postSpeechSamples = 0;
}

function dispatchForTranscriptionAndResetAudioBuffer(overflow) {
  const overflowLength = overflow?.length ?? 0;
  const buffer = BUFFER.slice(0, bufferPointer + SPEECH_PAD_SAMPLES);

  const prevLength = prevBuffers.reduce((acc, b) => acc + b.length, 0);
  const paddedBuffer = new Float32Array(prevLength + buffer.length);
  let offset = 0;
  for (const prev of prevBuffers) {
    paddedBuffer.set(prev, offset);
    offset += prev.length;
  }
  paddedBuffer.set(buffer, offset);

  speechToSpeech(paddedBuffer).catch((error) => {
    self.postMessage({ type: "voice_error", error: String(error) });
  });

  if (overflow) {
    BUFFER.set(overflow, 0);
  }
  resetAfterRecording(overflowLength);
}

async function speechToSpeech(buffer) {
  isPlaying = true;

  const text = await transcriber(buffer).then(({ text }) => text.trim());
  if (["", "[BLANK_AUDIO]"].includes(text)) {
    isPlaying = false;
    return;
  }

  self.postMessage({ type: "voice_transcript", text });
  messages.push({ role: "user", content: text });

  const splitter = new TextSplitterStream();
  const stream = tts.stream(splitter, { voice });

  (async () => {
    for await (const { text: chunkText, audio } of stream) {
      self.postMessage({ type: "voice_output_chunk", text: chunkText, audio });
    }
  })();

  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
  });

  let assistantFullText = "";
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (textChunk) => {
      assistantFullText += textChunk;
      splitter.push(textChunk);
      self.postMessage({ type: "voice_assistant_text", text: assistantFullText });
    },
  });

  stopping_criteria = new InterruptableStoppingCriteria();
  const { past_key_values, sequences } = await llm.generate({
    ...inputs,
    past_key_values: past_key_values_cache,
    do_sample: true,
    temperature: 0.8,
    top_p: 0.9,
    max_new_tokens: 196,
    streamer,
    stopping_criteria,
    return_dict_in_generate: true,
  });
  past_key_values_cache = past_key_values;

  splitter.close();

  const decoded = tokenizer.batch_decode(
    sequences.slice(null, [inputs.input_ids.dims[1], null]),
    { skip_special_tokens: true },
  );
  messages.push({ role: "assistant", content: decoded[0] });
}

self.onmessage = async (event) => {
  const { type, buffer } = event.data;

  if (type === "set_voice") {
    voice = event.data.voice || voice;
    return;
  }
  if (type === "interrupt") {
    stopping_criteria?.interrupt();
    return;
  }
  if (type === "playback_ended") {
    isPlaying = false;
    postStatus("idle", "Listening");
    return;
  }
  if (type === "reset_conversation") {
    messages = [SYSTEM_MESSAGE];
    past_key_values_cache = null;
    return;
  }

  if (type !== "audio") return;
  if (!silero_vad || !transcriber || !llm || isPlaying) return;

  const wasRecording = isRecording;
  const isSpeech = await vad(buffer);

  if (!wasRecording && !isSpeech) {
    if (prevBuffers.length >= MAX_NUM_PREV_BUFFERS) prevBuffers.shift();
    prevBuffers.push(buffer);
    return;
  }

  const remaining = BUFFER.length - bufferPointer;
  if (buffer.length >= remaining) {
    BUFFER.set(buffer.subarray(0, remaining), bufferPointer);
    bufferPointer += remaining;
    const overflow = buffer.subarray(remaining);
    dispatchForTranscriptionAndResetAudioBuffer(overflow);
    return;
  }

  BUFFER.set(buffer, bufferPointer);
  bufferPointer += buffer.length;

  if (isSpeech) {
    if (!isRecording) {
      postStatus("recording_start", "Listening...");
    }
    isRecording = true;
    postSpeechSamples = 0;
    return;
  }

  postSpeechSamples += buffer.length;

  if (postSpeechSamples < MIN_SILENCE_DURATION_SAMPLES) {
    return;
  }

  if (bufferPointer < MIN_SPEECH_DURATION_SAMPLES) {
    resetAfterRecording();
    return;
  }

  dispatchForTranscriptionAndResetAudioBuffer();
};

initModels().catch((error) => {
  self.postMessage({ type: "voice_error", error: String(error) });
});
