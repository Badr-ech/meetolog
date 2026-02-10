/** @type {import('next').NextConfig} */
const nextConfig = {
  // Set turbopack root to silence lockfile warning
  turbopack: {
    root: __dirname,
  },
  
  // Rewrite API calls to backend to avoid CORS issues
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
  
  // Increase timeouts for long-running transcription jobs
  experimental: {
    proxyTimeout: 300000, // 5 minutes for audio processing
  },
};

module.exports = nextConfig;
