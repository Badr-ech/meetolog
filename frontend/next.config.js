/** @type {import('next').NextConfig} */
const nextConfig = {
  // Rewrite API calls to backend to avoid CORS issues
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
