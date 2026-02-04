import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Meetolog - Meeting to Backlog",
  description: "Transform meeting recordings into structured Agile artifacts",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
