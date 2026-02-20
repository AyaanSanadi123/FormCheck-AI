// frontend/components/custom/KinematicsChart.tsx
import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, ReferenceLine 
} from 'recharts';

export function KinematicsChart({ fullData, activeTime }: { fullData: any[], activeTime: number }) {
  // 1. Sanitize the data: Only keep frames that actually have metrics
  const chartData = fullData
    .filter(frame => frame.rep_logic && frame.rep_logic.metrics && frame.rep_logic.metrics.elbow_angle !== undefined)
    .map(frame => ({
      time: frame.timestamp,
      elbow_angle: frame.rep_logic.metrics.elbow_angle,
      state: frame.rep_logic.state
    }));

  // Edge Case: The Calibration Void (No metrics yet)
  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full w-full bg-zinc-950 rounded-md border border-zinc-800 border-dashed">
        <p className="text-zinc-500 font-mono text-sm">AWAITING KINEMATIC DATA...</p>
      </div>
    );
  }

  // Custom Tooltip to show the AI State (e.g., ECCENTRIC) on hover
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-zinc-900 border border-zinc-700 p-3 rounded shadow-lg text-sm">
          <p className="text-zinc-400 mb-1">Time: {label}s</p>
          <p className="text-blue-400 font-bold">Angle: {payload[0].value}°</p>
          <p className="text-zinc-300">State: {payload[0].payload.state}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-full w-full bg-zinc-900 rounded-md p-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          
          <XAxis 
            dataKey="time" 
            stroke="#71717a" 
            tick={{ fill: '#71717a', fontSize: 12 }} 
            tickFormatter={(val) => `${val}s`} 
          />
          
          <YAxis 
            domain={['auto', 'auto']} 
            stroke="#71717a" 
            tick={{ fill: '#71717a', fontSize: 12 }} 
          />
          
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#3f3f46', strokeWidth: 1 }} />
          
          {/* The Data Line */}
          <Line 
            type="monotone" 
            dataKey="elbow_angle" 
            stroke="#3b82f6" 
            strokeWidth={3} 
            dot={false} 
            activeDot={{ r: 6, fill: '#3b82f6', stroke: '#fff' }} 
          />
          
          {/* THE PLAYHEAD TRACKER: Moves automatically with the video */}
          <ReferenceLine x={activeTime} stroke="#ef4444" strokeWidth={2} />
          
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}