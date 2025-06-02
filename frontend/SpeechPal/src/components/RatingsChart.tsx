import React, { useEffect, useRef } from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Bar } from 'react-chartjs-2';
// Import Chart.js core and necessary components to register them.
// This is crucial for Chart.js to work correctly.
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register the necessary Chart.js components.
// If you don't register them, the chart won't render.
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface RatingsChartProps {
  segments: Array<{ rating: number; excerpt: string }>;
}

export const RatingsChart: React.FC<RatingsChartProps> = ({ segments }) => {
  // useRef to hold the Chart.js instance.
  const chartRef = useRef<ChartJS | null>(null);

  // useEffect for cleanup: Destroy the chart when the component unmounts
  // or when the 'segments' prop changes (due to the key prop in parent).
  useEffect(() => {
    return () => {
      // On unmount or before re-rendering due to 'key' change in parent,
      // destroy the existing chart instance to free up the canvas.
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null; // Clear the ref
      }
    };
  }, [segments]); // Dependency array ensures this effect runs when segments change.
                  // The key prop in the parent component forces a remount,
                  // triggering this cleanup.

  if (!segments || segments.length === 0) return null;

  const data = {
    labels: segments.map((_, i) => `S${i + 1}`),
    datasets: [
      {
        label: 'AI Rating',
        data: segments.map(seg => seg.rating),
        backgroundColor: '#1976d2',
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false, // Allows the chart to take up the full height of its container
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (ctx: any) => `Segment ${ctx[0].label}`,
          label: (ctx: any) => `Rating: ${ctx.parsed.y}`,
          afterBody: (ctx: any) => {
            const idx = ctx[0].dataIndex;
            return segments[idx].excerpt ? `Excerpt: ${segments[idx].excerpt}` : '';
          },
        },
      },
      title: {
        display: true,
        text: 'AI Ratings Overview', // Added a chart title for clarity
      },
    },
    scales: {
      y: {
        min: 0,
        max: 10,
        title: { display: true, text: 'Rating' },
        ticks: { stepSize: 1 },
      },
      x: { // Added x-axis configuration for better readability
        title: { display: true, text: 'Segments' },
      }
    },
  };

  return (
    <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
      {/* Moved Typography for title outside the chart area if desired, or keep as plugin title */}
      {/* <Typography variant="subtitle1" sx={{ mb: 2 }}>AI Ratings Overview</Typography> */}
      <Box sx={{ height: 300 }}>
        <Bar
          // Store chart instance in the ref for cleanup
          ref={(el) => {
            if (el && el.chart) {
              chartRef.current = el.chart;
            }
          }}
          data={data}
          options={options}
        />
      </Box>
    </Paper>
  );
};