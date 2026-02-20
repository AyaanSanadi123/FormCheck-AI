// frontend/components/custom/CanonicalViewer.tsx
import React from "react";

// The standard connections for a 2D side-profile stick figure
const BONES = [
  ["shoulder", "elbow"],
  ["elbow", "wrist"],
  ["shoulder", "hip"],
  ["hip", "knee"],
  ["knee", "ankle"],
];

// Canvas scaling constants to handle Edge Case 2
const SCALE = 150; 
const ORIGIN_X = 200; // Center of a 400px wide SVG
const ORIGIN_Y = 150; // Placed near the top since the shoulder is usually (0,0)

export function CanonicalViewer({ frameData }: { frameData: any }) {
  // Edge Case 4: The "No Data" Void (Calibrating or missing data)
  if (!frameData || frameData.gatekeeper?.status === "CALIBRATING" || !frameData.normalized_coords) {
    return (
      <div className="flex items-center justify-center h-full w-full bg-zinc-950 rounded-md border border-zinc-800 border-dashed">
        <div className="flex flex-col items-center animate-pulse">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-zinc-400 font-mono text-sm tracking-wider">CALIBRATING COORDINATE SPACE...</p>
        </div>
      </div>
    );
  }

  const coords = frameData.normalized_coords;

  // Edge Case 1: The SVG Y-Axis Trap (Math inversion)
  const mapToSVG = (point: { x: number; y: number }) => ({
    x: ORIGIN_X + point.x * SCALE,
    y: ORIGIN_Y - point.y * SCALE, // The crucial minus sign to flip the Y-axis!
  });

  return (
    <div className="flex items-center justify-center h-full w-full bg-zinc-900 rounded-md overflow-hidden relative">
      {/* Background Grid for visual reference */}
      <div className="absolute inset-0" style={{
        backgroundImage: 'linear-gradient(#27272a 1px, transparent 1px), linear-gradient(90deg, #27272a 1px, transparent 1px)',
        backgroundSize: '20px 20px',
        opacity: 0.5
      }}></div>

      <svg width="400" height="500" className="relative z-10 overflow-visible">
        {/* Draw the Origin (0,0) axes lines */}
        <line x1="0" y1={ORIGIN_Y} x2="400" y2={ORIGIN_Y} stroke="#3f3f46" strokeWidth="1" strokeDasharray="4" />
        <line x1={ORIGIN_X} y1="0" x2={ORIGIN_X} y2="500" stroke="#3f3f46" strokeWidth="1" strokeDasharray="4" />

        {/* Edge Case 3: The Ghost Joint (Safe Bone Drawing) */}
        {BONES.map(([jointA, jointB]) => {
          const ptA = coords[jointA];
          const ptB = coords[jointB];

          // Only draw if both points exist in the JSON payload
          if (!ptA || !ptB) return null;

          const svgA = mapToSVG(ptA);
          const svgB = mapToSVG(ptB);

          return (
            <line
              key={`${jointA}-${jointB}`}
              x1={svgA.x} y1={svgA.y}
              x2={svgB.x} y2={svgB.y}
              stroke="#3b82f6" // Blue bones
              strokeWidth="4"
              strokeLinecap="round"
            />
          );
        })}

        {/* Draw the Joint Dots on top of the bones */}
        {Object.entries(coords).map(([jointName, point]: [string, any]) => {
          if (!point) return null;
          const svgPt = mapToSVG(point);
          
          return (
            <circle
              key={jointName}
              cx={svgPt.x}
              cy={svgPt.y}
              r="6"
              fill="#10b981" // Green joints
              stroke="#ffffff"
              strokeWidth="2"
            />
          );
        })}
      </svg>
    </div>
  );
}