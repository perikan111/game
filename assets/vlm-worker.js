import {
  AutoProcessor,
  AutoModelForVision2Seq,
  TextStreamer,
  InterruptableStoppingCriteria,
  load_image,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1";

// HuggingFaceTB/SmolVLM-Instruct: the original 2B-parameter SmolVLM checkpoint.
// The decoder (the bulk of the 2B params) is quantized to 4-bit; the vision
// encoder is kept at fp16 since encoder-decoder VLMs are sensitive to
// quantizing the vision tower.
const MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct";
const MAX_NEW_TOKENS = 128;

async function check() {
  try {
    if (!self.navigator.gpu) {
      throw new Error("このブラウザは WebGPU に対応していません。");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU アダプタが見つかりませんでした。");
    }
  } catch (e) {
    self.postMessage({ status: "error", data: e.toString() });
  }
}

class VLM {
  static processor;
  static model;

  static async getInstance(progress_callback = null) {
    this.processor ??= AutoProcessor.from_pretrained(MODEL_ID, {
      progress_callback,
    });

    this.model ??= AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
      dtype: {
        embed_tokens: "fp16",
        vision_encoder: "fp16",
        decoder_model_merged: "q4",
      },
      device: "webgpu",
      progress_callback,
    });

    return Promise.all([this.processor, this.model]);
  }
}

const stopping_criteria = new InterruptableStoppingCriteria();

async function load() {
  self.postMessage({ status: "loading", data: "モデルを読み込んでいます…" });
  try {
    await VLM.getInstance((x) => self.postMessage(x));
    self.postMessage({ status: "ready" });
  } catch (e) {
    // Drop the cached (rejected) promises so the next "load" retry actually
    // re-fetches instead of immediately re-throwing the same failure.
    VLM.processor = undefined;
    VLM.model = undefined;
    self.postMessage({ status: "error", data: e.toString() });
  }
}

async function generate({ image, prompt }) {
  self.postMessage({ status: "start" });

  // Everything below can throw (bad frame, processor/template issues, OOM,
  // WebGPU buffer errors, ...). It all has to be caught here, otherwise a
  // rejection just vanishes and the page is left waiting forever with no
  // "error" message ever posted back.
  try {
    const [processor, model] = await VLM.getInstance();

    const messages = [
      { role: "user", content: [{ type: "image", image }, { type: "text", text: prompt }] },
    ];

    const images = await Promise.all(
      messages
        .map((x) => x.content)
        .flat(Infinity)
        .filter((msg) => msg.image !== undefined)
        .map((msg) => load_image(msg.image)),
    );

    const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
    const inputs = await processor(text, images);

    let startTime;
    let numTokens = 0;
    let tps = 0;
    const token_callback_function = () => {
      startTime ??= performance.now();
      if (numTokens++ > 0) {
        tps = (numTokens / (performance.now() - startTime)) * 1000;
      }
    };
    const callback_function = (output) => {
      self.postMessage({ status: "update", output, tps, numTokens });
    };

    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function,
      token_callback_function,
    });

    await model.generate({
      ...inputs,
      do_sample: false,
      repetition_penalty: 1.1,
      max_new_tokens: MAX_NEW_TOKENS,
      streamer,
      stopping_criteria,
    });

    self.postMessage({ status: "complete" });
  } catch (e) {
    self.postMessage({ status: "error", data: e.toString() });
  }
}

self.addEventListener("error", (e) => {
  self.postMessage({ status: "error", data: `${e.message} (${e.filename}:${e.lineno})` });
});
self.addEventListener("unhandledrejection", (e) => {
  self.postMessage({ status: "error", data: "Unhandled: " + e.reason });
});

self.addEventListener("message", async (e) => {
  const { type, data } = e.data;
  switch (type) {
    case "check":
      check();
      break;
    case "load":
      load();
      break;
    case "generate":
      stopping_criteria.reset();
      generate(data);
      break;
    case "interrupt":
      stopping_criteria.interrupt();
      break;
  }
});
