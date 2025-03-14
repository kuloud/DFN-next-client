'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Input, Button, Card, Progress, Textarea } from "@nextui-org/react";
import { NextUIProvider } from "@nextui-org/react";

export default function Home() {

  const [result, setResult] = useState(null);
  const [ready, setReady] = useState(null);
  const [error, setError] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [urlInput, setUrlInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const [downloads, setDownloads] = useState(new Map());

  const [completedDownloads, setCompletedDownloads] = useState(new Set());

  // Worker 引用
  const worker = useRef(null);

  useEffect(() => {
    if (!worker.current) {
      worker.current = new Worker(new URL('./worker.ts', import.meta.url), {
        type: 'module'
      });
    }

    const onMessageReceived = (e) => {
      console.log('Message received:', e.data);
      switch (e.data.status) {
        case 'initiate':
          setReady(false);
          setIsLoading(true);
          setDownloads(new Map());
          setCompletedDownloads(new Set());
          break;
        case 'progress':
          const progressData = e.data.progress;
          // Only update progress if it's a real download (not from cache)
          if (progressData.total > 0) {
            setDownloads(prev => {
              const newDownloads = new Map(prev);
              newDownloads.set(progressData.file, {
                name: progressData.name,
                progress: progressData.progress,
                loaded: progressData.loaded,
                total: progressData.total
              });
              return newDownloads;
            });
            if (progressData.progress === 100) {
              setCompletedDownloads(prev => new Set(prev).add(progressData.file));
            }
          }
          break;
        case 'ready':
          setReady(true);
          setIsLoading(false);
          // Clear downloads display when ready
          setDownloads(new Map());
          break;
        case 'complete':
          setResult(e.data.output);
          setIsLoading(false);
          break;
        case 'error':
          setError(e.data.error);
          setIsLoading(false);
          break;
      }
    };

    worker.current.addEventListener('message', onMessageReceived);
    return () => worker.current.removeEventListener('message', onMessageReceived);
  });

  const submit = useCallback((text, url) => {
    if (worker.current) {
      setError(null);
      setIsLoading(true);
      worker.current.postMessage({ 
        text: text.trim(),
        url: url.trim() 
      });
    }
  }, []);

  // Calculate button disabled state
  const isSubmitDisabled = isLoading || 
    !(textInput.trim() && urlInput.trim()); // Require both inputs

  // Format bytes to human readable format
  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  // Get filename from path
  const getFileName = (path) => {
    return path ? path.split('/').pop() : '';
  };

  return (
    <NextUIProvider>
      <main className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-br from-blue-50 to-violet-50">
        <Card className="w-full max-w-3xl p-6 flex flex-col">

          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
              Transformers.js
            </h1>
          </div>

          {/* Input Section */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="col-span-2 md:col-span-1">
              <Textarea
                label="Text"
                placeholder="Enter text to analyze..."
                value={textInput}
                onValueChange={setTextInput}
                variant="bordered"
                classNames={{
                  input: "resize-none",
                  label: "text-sm font-medium",
                }}
              />
            </div>
            <div className="col-span-2 md:col-span-1">
              <Textarea
                label="URL"
                placeholder="Enter URL to analyze..."
                value={urlInput}
                onValueChange={setUrlInput}
                variant="bordered"
                classNames={{
                  input: "resize-none",
                  label: "text-sm font-medium",
                }}
              />
            </div>
          </div>

          {/* Action Button */}
          <div className="flex justify-center mb-6">
            <Button
              color="primary"
              size="lg"
              className="px-12 font-medium"
              onClick={() => submit(textInput, urlInput)}
              isDisabled={isSubmitDisabled}
              isLoading={isLoading}
              startContent={!isLoading && <IconAnalyze />}
            >
              Submit
            </Button>
          </div>

          {/* Status and Results */}
          <div className="space-y-4">
            {/* Progress Display */}
            {downloads.size > 0 && Array.from(downloads.values()).some(d => d.total > 0) && (
              <Card className="p-4 bg-default-50">
                <div className="space-y-3">
                  {Array.from(downloads.entries())
                    .filter(([_, data]) => data.total > 0) // Only show real downloads
                    .map(([file, data]) => (
                      <div key={file} className="space-y-2">
                        <div className="flex justify-between text-sm text-default-600">
                          <span className="font-medium">
                            {getFileName(file)}
                            {completedDownloads.has(file) && 
                              <span className="ml-2 text-success">✓</span>
                            }
                          </span>
                          <span>
                            {formatBytes(data.loaded)} / {formatBytes(data.total)}
                          </span>
                        </div>
                        {!completedDownloads.has(file) && (
                          <Progress
                            value={data.progress}
                            color="primary"
                            size="sm"
                            showValueLabel={true}
                            classNames={{
                              value: "text-sm font-medium",
                            }}
                          />
                        )}
                      </div>
                    ))}
                </div>
              </Card>
            )}

            {/* Error Message */}
            {error && (
              <Card className="p-4 bg-danger-50">
                <p className="text-danger text-sm">{error}</p>
              </Card>
            )}

            {/* Results Display */}
            {result !== null && (
              <Card className="p-4 bg-default-50">
                <div className="space-y-2">
                  <div className="text-center text-xl font-medium">
                    Similarity Score
                  </div>
                  <div className="text-center text-3xl font-bold text-primary">
                    {(result * 100).toFixed(1)}%
                  </div>
                  <div className="text-center text-sm text-default-500">
                    {result < 0.3 ? 'Low Match' : 
                     result < 0.6 ? 'Moderate Match' : 
                     'High Match'}
                  </div>
                </div>
              </Card>
            )}
          </div>
        </Card>
      </main>
    </NextUIProvider>
  )
}

// Analysis Icon Component
const IconAnalyze = () => (
  <svg 
    width="20" 
    height="20" 
    viewBox="0 0 24 24" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <path 
      d="M20 12V17C20 18.8856 20 19.8284 19.4142 20.4142C18.8284 21 17.8856 21 16 21H8C6.11438 21 5.17157 21 4.58579 20.4142C4 19.8284 4 18.8856 4 17V7C4 5.11438 4 4.17157 4.58579 3.58579C5.17157 3 6.11438 3 8 3H13M15 3H19M19 3V7M19 3L11 11" 
      stroke="currentColor" 
      strokeWidth="2" 
      strokeLinecap="round" 
      strokeLinejoin="round"
    />
  </svg>
)
