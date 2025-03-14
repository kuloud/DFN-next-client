# Text-Image Similarity Analysis

A web application that analyzes the semantic similarity between text and images using the DFN (Data Filtering Networks) model. Built with Next.js, Transformers.js, and NextUI.

![loading models](https://github.com/kuloud/DFN-next-client/blob/main/screenshots/image.png)
![result](https://github.com/kuloud/DFN-next-client/blob/main/screenshots/image1.png)

## Features

- Real-time text and image URL similarity analysis
- Client-side model processing using Web Workers
- Progressive model loading with status indicators
- Modern UI with responsive design
- Browser-based inference without server requirements

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **UI Library**: NextUI
- **ML Model**: Transformers.js
- **Model**: DFN-public (CLIP-based Data Filtering Networks)
- **Styling**: Tailwind CSS

## Getting Started

### Prerequisites

- Node.js 18+ 
- pnpm (recommended) or npm

### Installation

1. Clone the repository
2. Navigate to the project directory
3. Install dependencies using pnpm or npm

```bash
pnpm install
# or
npm install
```

## Usage

1. Enter text in the "Text" input field
2. Provide an image URL in the "URL" input field
3. Click "Submit" to analyze
4. View the similarity score and match level

The application will calculate a similarity score between 0-100%, indicating how well the text semantically matches the image.

## Technical Details

### Model Information

- Uses the DFN (Data Filtering Networks) model
- Processes text and images in parallel
- Computes cosine similarity between text and image embeddings
- Model is loaded and cached in the browser

### Performance

- Client-side processing using Web Workers
- Progressive model downloading with size indicators
- Caches model for subsequent uses
- FP16 precision for optimal performance

## Browser Compatibility

- Modern browsers with WebAssembly support
- Recommended: Chrome, Firefox, Safari, Edge (latest versions)

## License

[Your chosen license]

## Acknowledgments

- [Transformers.js](https://huggingface.co/docs/transformers.js) by Hugging Face
- [NextUI](https://nextui.org/) for the UI components
- [DFN Model](https://huggingface.co/XudongShen/DFN-public) by XudongShen
