import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Bar } from 'react-chartjs-2';

interface RatingsChartProps {
  segments: Array<{ rating: number; excerpt: string }>;

}

export const RatingsChart: React.FC<RatingsChartProps> = ({ segments }) => {
  if (!segments || segments.length === 0) return null;

  const data = {
    labels: segments.map((seg, i) => `S${i + 1}`),
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
    },
    scales: {
      y: {
        min: 0,
        max: 10,
        title: { display: true, text: 'Rating' },
        ticks: { stepSize: 1 },
      },
    },
  };

  return (
    <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>AI Ratings Overview</Typography>
      <Box sx={{ height: 300 }}>
        <Bar data={data} options={options} />
      </Box>
    </Paper>
  );
};