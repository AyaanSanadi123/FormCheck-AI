"use client";

import { useState, useEffect, useRef } from "react";
import { CanonicalViewer } from "@/components/custom/CanonicalViewer";
import { KinematicsChart } from "@/components/custom/KinematicsChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// NEW SHADCN IMPORTS
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";

export default function TestingStudio() {
  const [telemetryData, setTelemetryData] = useState<any[]>([]);
  const [activeFrameIdx, setActiveFrameIdx] = useState<number>(0);
  
  const [driveUrl, setDriveUrl] = useState<string>("");
  const [exerciseName, setExerciseName] = useState<string>("dips");
  const [statusMsg, setStatusMsg] = useState<string>("Idle");
  
  // States for Video Playback
  const [videoUrl, setVideoUrl] = useState<string>(""); 
  const [processedCount, setProcessedCount] = useState<number>(0);
  const [currentVideoId, setCurrentVideoId] = useState<number>(1);
  const videoRef = useRef<HTMLVideoElement>(null);

  // --- NEW: PAGINATION & SCANNING STATES ---
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isClearing, setIsClearing] = useState<boolean>(false);
  
  const [totalVideos, setTotalVideos] = useState<number>(0);
  const [batchSize, setBatchSize] = useState<number>(5);
  const [selectedBatchIndex, setSelectedBatchIndex] = useState<string>("0");
  const [showOverwriteModal, setShowOverwriteModal] = useState<boolean>(false);

  // Reset scan if URL changes
  useEffect(() => {
    setTotalVideos(0);
  }, [driveUrl]);

  // --- 1. THE SCOUT (Peek Folder) ---
  const scanFolder = async () => {
    if (!driveUrl) return alert("Please paste a Google Drive folder link.");
    
    setIsScanning(true);
    setStatusMsg("Scanning Google Drive folder...");
    
    try {
      const res = await fetch("http://localhost:8000/api/test/peek", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drive_folder_url: driveUrl })
      });
      
      if (!res.ok) throw new Error("Failed to peek into folder.");
      
      const data = await res.json();
      setTotalVideos(data.data.total_videos);
      setBatchSize(data.data.batch_size);
      setSelectedBatchIndex("0"); // Default to Batch 1
      setStatusMsg(`Found ${data.data.total_videos} videos. Ready to process.`);
    } catch (error: any) {
      console.error(error);
      setStatusMsg(`Scan Error: ${error.message}`);
    } finally {
      setIsScanning(false);
    }
  };

  // --- 2. THE SAFETY INTERCEPTOR ---
  const handleProcessClick = () => {
    if (processedCount > 0) {
      // Catch them before they overwrite existing data!
      setShowOverwriteModal(true);
    } else {
      executePipeline(Number(selectedBatchIndex));
    }
  };

  // --- 3. THE EXECUTION LOGIC ---
  const executePipeline = async (batchIdx: number) => {
    setIsProcessing(true);
    setStatusMsg(`Downloading & processing Batch ${batchIdx + 1}...`);
    
    // Clear the UI while loading
    setTelemetryData([]); 
    setVideoUrl("");
    setProcessedCount(0);

    try {
      const triggerRes = await fetch("http://localhost:8000/api/test/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          drive_folder_url: driveUrl, 
          exercise_name: exerciseName,
          batch_index: batchIdx
        })
      });

      if (!triggerRes.ok) throw new Error("Backend API failed to start the job.");
      
      const triggerData = await triggerRes.json();

      // --- NEW: Check if the backend caught a fatal exception ---
      if (triggerData.data?.critical_error) {
        throw new Error(`Backend Exception: ${triggerData.data.critical_error}`);
      }

      const successCount = triggerData.data?.successful || 0;
      setProcessedCount(successCount);

      if (successCount > 0) {
        await loadVideoData(1);
      } else {
        setStatusMsg("Pipeline finished, but no videos were successfully processed.");
      }
    } catch (error: any) {
      console.error(error);
      setStatusMsg(`Pipeline Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const loadVideoData = async (videoId: number) => {
    try {
      setStatusMsg(`Loading Video ${videoId} telemetry...`);
      const telemetryRes = await fetch(`http://localhost:8000/api/test/results/${videoId}`);
      if (!telemetryRes.ok) throw new Error(`Failed to fetch JSON for Video ${videoId}.`);
      
      const data = await telemetryRes.json();
      setTelemetryData(data);
      
      // FIX 1: Add a timestamp to bypass aggressive browser caching
      setVideoUrl(`http://localhost:8000/media/${videoId}.mp4?t=${Date.now()}`);
      
      setCurrentVideoId(videoId);
      setActiveFrameIdx(0); 
      setStatusMsg(`Video ${videoId} loaded successfully!`);
    } catch (error: any) {
      console.error(error);
      setStatusMsg(`Error: ${error.message}`);
    }
  };

  const sweepCache = async () => {
    if (!confirm("Are you sure? This will delete all downloaded videos and telemetry from your hard drive.")) return;
    setIsClearing(true);
    setStatusMsg("Sweeping backend cache...");
    try {
      const res = await fetch("http://localhost:8000/api/test/cache", { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to clear backend cache.");
      setTelemetryData([]);
      setVideoUrl("");
      setProcessedCount(0);
      setCurrentVideoId(1);
      setStatusMsg("Cache successfully swept. Ready for new ingestion.");
    } catch (error: any) {
      setStatusMsg(`Error clearing cache: ${error.message}`);
    } finally {
      setIsClearing(false);
    }
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current || telemetryData.length === 0) return;
    const currentTime = videoRef.current.currentTime;
    let newFrameIdx = 0;
    for (let i = telemetryData.length - 1; i >= 0; i--) {
      if (telemetryData[i].timestamp <= currentTime) {
        newFrameIdx = i;
        break;
      }
    }
    if (newFrameIdx !== activeFrameIdx) {
      setActiveFrameIdx(newFrameIdx);
    }
  };

  const activeFrameData = telemetryData[activeFrameIdx] || null;

  // Calculate how many batches exist
  const numBatches = Math.ceil(totalVideos / batchSize);

  return (
    <div className="min-h-screen bg-zinc-950 text-white p-6">
      
      {/* THE SAFETY MODAL */}
      <AlertDialog open={showOverwriteModal} onOpenChange={setShowOverwriteModal}>
        <AlertDialogContent className="bg-zinc-900 border border-zinc-800 text-white">
          <AlertDialogHeader>
            <AlertDialogTitle>Overwrite Local Cache?</AlertDialogTitle>
            <AlertDialogDescription className="text-zinc-400">
              You currently have processed videos sitting in your local testing cache. Loading a new batch will delete them to free up space. Do you want to proceed?
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-zinc-800 text-white hover:bg-zinc-700 border-none">Cancel</AlertDialogCancel>
            <AlertDialogAction 
              className="bg-red-600 hover:bg-red-500 text-white"
              onClick={() => {
                setShowOverwriteModal(false);
                executePipeline(Number(selectedBatchIndex));
              }}
            >
              Yes, Overwrite
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <header className="mb-4 flex flex-col lg:flex-row justify-between lg:items-end gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">FormCheck-AI: Debug Studio</h1>
          <p className="text-zinc-400">Time-travel video telemetry & kinematics analysis</p>
        </div>

        {/* --- DYNAMIC CONTROL PANEL --- */}
        <div className="flex flex-col gap-3">
          
          {/* Row 1: The Scanner */}
          <div className="flex flex-col sm:flex-row gap-3 bg-zinc-900 p-3 rounded-lg border border-zinc-800">
            <input 
              suppressHydrationWarning
              type="text" 
              placeholder="Google Drive Folder URL" 
              value={driveUrl}
              onChange={(e) => setDriveUrl(e.target.value)}
              className="bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm w-full sm:w-64 focus:outline-none focus:border-blue-500"
            />
            <input 
              suppressHydrationWarning
              type="text" 
              placeholder="Exercise (e.g. dips)" 
              value={exerciseName}
              onChange={(e) => setExerciseName(e.target.value)}
              className="bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm w-full sm:w-32 focus:outline-none focus:border-blue-500"
            />
            <button 
              suppressHydrationWarning
              onClick={scanFolder}
              disabled={isScanning}
              className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded text-sm font-bold transition-colors disabled:opacity-50"
            >
              {isScanning ? "Scanning..." : "Scan Folder"}
            </button>
          </div>

          {/* Row 2: The Batch Selector (Only shows if videos are found) */}
          {totalVideos > 0 && (
            <div className="flex flex-col sm:flex-row gap-3 bg-blue-950/20 p-3 rounded-lg border border-blue-900/50 items-center justify-between">
              <div className="flex items-center gap-3 w-full">
                <span className="text-sm text-blue-400 font-bold whitespace-nowrap">Select Batch:</span>
                <Select value={selectedBatchIndex} onValueChange={setSelectedBatchIndex}>
                  <SelectTrigger className="w-full sm:w-64 bg-zinc-950 border-zinc-700">
                    <SelectValue placeholder="Select a batch" />
                  </SelectTrigger>
                  <SelectContent className="bg-zinc-900 border-zinc-700 text-white">
                    {Array.from({ length: numBatches }).map((_, i) => {
                      const startVid = i * batchSize + 1;
                      const endVid = Math.min((i + 1) * batchSize, totalVideos);
                      return (
                        <SelectItem key={i} value={i.toString()} className="hover:bg-zinc-800 focus:bg-zinc-800 cursor-pointer">
                          Batch {i + 1} (Videos {startVid} - {endVid})
                        </SelectItem>
                      );
                    })}
                  </SelectContent>
                </Select>
              </div>
              
              <button 
                onClick={handleProcessClick}
                disabled={isProcessing}
                className={`px-4 py-2 w-full sm:w-auto rounded text-sm font-bold transition-colors whitespace-nowrap ${
                  isProcessing ? "bg-zinc-700 text-zinc-500 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-500 text-white"
                }`}
              >
                {isProcessing ? "Processing..." : "Process Batch"}
              </button>
            </div>
          )}

        </div>
      </header>

      {/* --- INTRA-BATCH NAVIGATOR & STATUS BAR --- */}
      <div className="mb-6 flex flex-col sm:flex-row justify-between items-center gap-4 bg-zinc-900 px-4 py-3 rounded border border-zinc-800">
        <div className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto">
          <div className="text-sm font-mono truncate w-full sm:w-auto max-w-[300px] lg:max-w-md">
            Status: <span className={statusMsg.includes("Error") ? "text-red-400" : "text-green-400"}>{statusMsg}</span>
          </div>
          
          {processedCount > 0 && (
            <div className="flex gap-2 sm:border-l sm:border-zinc-700 sm:pl-4">
              <span className="text-sm text-zinc-400 flex items-center mr-2 hidden sm:flex">Videos:</span>
              {Array.from({ length: processedCount }).map((_, index) => {
                const vidId = index + 1;
                return (
                  <button
                    key={vidId}
                    onClick={() => loadVideoData(vidId)}
                    className={`px-3 py-1 text-sm rounded transition-colors ${
                      currentVideoId === vidId 
                        ? "bg-blue-600 text-white font-bold" 
                        : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                    }`}
                  >
                    {vidId}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <button 
          onClick={sweepCache}
          disabled={isClearing || processedCount === 0}
          className={`px-4 py-1.5 text-sm font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${
            isClearing || processedCount === 0 
              ? "bg-red-900/20 text-red-500/50 cursor-not-allowed border border-red-900/30" 
              : "bg-red-950/40 text-red-500 hover:bg-red-900/60 border border-red-900/50"
          }`}
        >
          {isClearing ? "Sweeping..." : "Sweep Cache"}
        </button>
      </div>

      {/* Main Split-Screen Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[85vh]">
        
        {/* LEFT COLUMN: Video & Controls */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <Card className="bg-zinc-900 border-zinc-800 flex-grow shadow-lg">
            <CardHeader className="py-4">
              <CardTitle className="text-lg">Video Playback</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="w-full aspect-video bg-black rounded-md border border-zinc-700 overflow-hidden relative shadow-inner">
                {videoUrl ? (
                  <video 
                    key={videoUrl} /* FIX 2: Force React to remount the player when the URL changes */
                    ref={videoRef}
                    src={videoUrl} 
                    controls
                    onTimeUpdate={handleTimeUpdate}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-zinc-600 font-mono text-xs tracking-widest">
                    AWAITING INGESTION
                  </div>
                )}
              </div>
              
              <div className="mt-6 flex justify-between items-center text-xs text-zinc-400 mb-2 font-mono uppercase tracking-wide">
                <p>Timeline Sync</p>
                <p>Frame {activeFrameData?.frame_id || 0}</p>
              </div>
              <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500 transition-all duration-75 ease-linear"
                  style={{ width: `${(activeFrameIdx / Math.max(1, telemetryData.length - 1)) * 100}%` }}
                ></div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* RIGHT COLUMN: The X-Ray Data Panels */}
        <div className="lg:col-span-7 flex flex-col">
          <Card className="bg-zinc-900 border-zinc-800 flex-grow shadow-lg">
            <CardHeader className="py-4">
              <CardTitle className="text-lg">Pipeline Telemetry</CardTitle>
            </CardHeader>
            <CardContent className="h-full pb-4">
              <Tabs defaultValue="canonical" className="w-full h-full flex flex-col">
                <TabsList className="grid w-full grid-cols-3 bg-zinc-950 border border-zinc-800">
                  <TabsTrigger value="canonical" className="data-[state=active]:bg-zinc-800">2D Canonical</TabsTrigger>
                  <TabsTrigger value="kinematics" className="data-[state=active]:bg-zinc-800">Kinematics</TabsTrigger>
                  <TabsTrigger value="raw" className="data-[state=active]:bg-zinc-800">Raw JSON</TabsTrigger>
                </TabsList>
                
                <div className="mt-4 flex-grow min-h-[500px] border border-zinc-800 rounded-md bg-black p-4 overflow-hidden relative">
                  <TabsContent value="canonical" className="h-full m-0 absolute inset-0 p-4">
                    <CanonicalViewer frameData={activeFrameData} />
                  </TabsContent>
                  <TabsContent value="kinematics" className="h-full m-0 absolute inset-0 p-4">
                    <KinematicsChart fullData={telemetryData} activeTime={activeFrameData?.timestamp || 0} />
                  </TabsContent>
                  <TabsContent value="raw" className="h-full m-0 absolute inset-0 p-4 overflow-auto">
                    <pre className="text-[11px] leading-tight text-green-400 font-mono">
                      {activeFrameData 
                        ? JSON.stringify(activeFrameData, null, 2) 
                        : "// Waiting for telemetry..."}
                    </pre>
                  </TabsContent>
                </div>
              </Tabs>
            </CardContent>
          </Card>
        </div>

      </div>
    </div>
  );
}