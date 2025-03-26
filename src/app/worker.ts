import { AutoProcessor, AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, FeatureExtractionPipeline, ImageFeatureExtractionPipeline, pipeline, PipelineType, PreTrainedModel, PreTrainedTokenizer, Processor, RawImage, Tensor } from "@huggingface/transformers";

class DFN {
  private processor?: Processor;
  private tokenizer?: PreTrainedTokenizer;
  private textModel?: PreTrainedModel;
  private visionModel?: PreTrainedModel;

  constructor() {}

  async initialize(modelName: string = 'XudongShen/DFN-public', progress_callback = null) {
    this.processor = await AutoProcessor.from_pretrained(modelName, {});
    this.tokenizer = await AutoTokenizer.from_pretrained(modelName);
    this.textModel = await CLIPTextModelWithProjection.from_pretrained(modelName, {
      progress_callback,
      dtype: "fp32",
      device: "cpu",
    });
    this.visionModel = await CLIPVisionModelWithProjection.from_pretrained(modelName, {
      progress_callback,
      dtype: "fp32",
      device: "cpu",
    });
    this.processor.image_processor.do_resize = false;
    return this;
  }

  getProcessor() {
    return this.processor;
  }

  getTokenizer() {
    return this.tokenizer;
  }

  getTextModel() {
    return this.textModel;
  }

  getVisionModel() {
    return this.visionModel;
  }
}

// Use the Singleton pattern to enable lazy construction of the pipeline.
class PipelineSingleton {
    static model = 'XudongShen/DFN-public';
    static instance: DFN | null = null;

    static async getInstance(progress_callback = null) {
        if (!this.instance) {
            self.postMessage({ status: 'initiate' });
            this.instance = await new DFN().initialize(this.model, (progress) => {
                // Only send progress for actual downloads
                if (progress.total > 0) {
                    self.postMessage({ status: 'progress', progress });
                }
            });
            self.postMessage({ status: 'ready' });
        }
        return this.instance;
    }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
    const { text, url } = event.data;
    const dfn = await PipelineSingleton.getInstance(x => self.postMessage({ status: 'progress', progress: x }));

    console.log("DFN initialized");
  
    function padToSquare(img: RawImage): RawImage {
      const width = img.width;
      const height = img.height;
      const maxDim = Math.max(width, height);

      const newImg = new RawImage(
        new Uint8Array(maxDim * maxDim * img.channels),
        maxDim,
        maxDim,
        img.channels
      );

      const fillColorValue = [0, 0, 0, 255];
      for (let y = 0; y < maxDim; y++) {
        for (let x = 0; x < maxDim; x++) {
          const index = (y * maxDim + x) * img.channels;
          newImg.data[index] = fillColorValue[0];
          newImg.data[index + 1] = fillColorValue[1];
          newImg.data[index + 2] = fillColorValue[2];
          if (img.channels === 4) {
            newImg.data[index + 3] = fillColorValue[3];
          }
        }
      }

      const offsetX = Math.floor((maxDim - width) / 2);
      const offsetY = Math.floor((maxDim - height) / 2);

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const srcIndex = (y * width + x) * img.channels;
          const destIndex = ((y + offsetY) * maxDim + (x + offsetX)) * img.channels;
          newImg.data[destIndex] = img.data[srcIndex];
          newImg.data[destIndex + 1] = img.data[srcIndex + 1];
          newImg.data[destIndex + 2] = img.data[srcIndex + 2];
          if (img.channels === 4) {
            newImg.data[destIndex + 3] = img.data[srcIndex + 3];
          }
        }
      }
  
      return newImg;
    }

    const cosineSimilarity = (vecA, vecB) => {
        let dot = 0.0;
        let normA = 0.0;
        let normB = 0.0;
    
        for (let i = 0; i < vecA.length; i++) {
          dot += vecA[i] * vecB[i];
          normA += vecA[i] * vecA[i];
          normB += vecB[i] * vecB[i];
        }
    
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
      }

      function normalize(vec: Float32Array): Float32Array {
        let norm = 0.0;
        for (let i = 0; i < vec.length; i++) {
          norm += vec[i] * vec[i];
        }
        norm = Math.sqrt(norm);
      
        const result = new Float32Array(vec.length);
        for (let i = 0; i < vec.length; i++) {
          result[i] = vec[i] / norm;
        }
      
        return result;
      }

    const computeSimilarity = async (text: string, imageUrl: string) => {
        if (!dfn) {
          throw new Error("DFN not initialized. Call initialize() first.");
        }
    
        try {
          console.log("Computing similarity...", text, imageUrl);
    
          // Get text embeddings
          // Get tokenizer max length
          const maxLength = dfn.getTokenizer().model_max_length; // default CLIP length is 77

          // Get complete token sequence without truncation
          const tokens = dfn.getTokenizer()([text], {
              truncation: false,
              padding: false
          }).input_ids.ort_tensor.cpuData;

          // Split into chunks
          const tokenChunks = [];
          for (let i = 0; i < tokens.length; i += maxLength) {
              tokenChunks.push(tokens.slice(i, i + maxLength));
          }

          // Process each chunk and calculate embeddings
          const chunkEmbeddings = [];
          for (const chunk of tokenChunks) {
            if (chunk.length === 0) {
                continue;
            }
            const chunkBigIntArray = Array.from(chunk, BigInt)
              // Convert tokens back to text
              const chunkText = dfn.getTokenizer().decode(chunkBigIntArray, {
                  skip_special_tokens: true
              });

              // Get chunk embedding
              const textInputs = dfn.getTokenizer()([chunkText], {
                  padding: "max_length",
                  truncation: true
              });
              const textOutputs = await dfn.getTextModel()(textInputs);
              const chunkEmbedding = normalize(textOutputs.text_embeds.ort_tensor.cpuData);
              chunkEmbeddings.push(chunkEmbedding);
          }

          // Calculate mean embedding
          let textEmbedding: Float32Array;
          if (chunkEmbeddings.length === 1) {
              textEmbedding = chunkEmbeddings[0];
          } else {
              // Calculate mean of all chunk embeddings
              textEmbedding = new Float32Array(chunkEmbeddings[0].length);
              for (const embedding of chunkEmbeddings) {
                  for (let i = 0; i < embedding.length; i++) {
                      textEmbedding[i] += embedding[i] / chunkEmbeddings.length;
                  }
              }
              // Renormalize mean embedding if multiple chunks
              textEmbedding = normalize(textEmbedding);
          }
          console.log('textEmbedding', textEmbedding)
    
          // Get image embeddings

          const originImage = await RawImage.read(imageUrl);
          const image = padToSquare(originImage);
          const imageInputs = await dfn.getProcessor()([image]);
          const imageOutputs = await dfn.getVisionModel()(imageInputs);
          const imageEmbedding = normalize( imageOutputs.image_embeds.ort_tensor.cpuData );
    
          // Compute cosine similarity
          return cosineSimilarity(textEmbedding, imageEmbedding);
        } catch (error) {
          console.error("Error computing similarity:", error);
          throw error;
        }
      }

    // Actually perform the classification
    const output = await computeSimilarity(text, url);

    console.log("Output", output);
    // Send the output back to the main thread
    self.postMessage({
        status: 'complete',
        output: output,
    });
});
