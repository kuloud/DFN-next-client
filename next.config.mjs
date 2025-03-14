/** @type {import('next').NextConfig} */
import path from "path";
import {fileURLToPath} from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const nextConfig = {
    // (Optional) Export as a static site
    // See https://nextjs.org/docs/pages/building-your-application/deploying/static-exports#configuration
    output: 'export', // Feel free to modify/remove this option
    images: {
        unoptimized: true,  // GitHub Pages 不支持 Next.js 的图片优化
    },
    basePath: '/DFN-next-client',

    // Override the default webpack configuration
    webpack: (config) => {
        // Ignore node-specific modules when bundling for the browser
        // See https://webpack.js.org/configuration/resolve/#resolvealias
        config.resolve.alias['@huggingface/transformers'] = path.resolve(__dirname, 'node_modules/@huggingface/transformers');
        config.resolve.alias = {
            ...config.resolve.alias,
            "sharp$": false,
            "onnxruntime-node$": false,
        }
        return config;
    },
    
};

export default nextConfig;
