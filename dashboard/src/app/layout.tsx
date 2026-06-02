import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Project Apex — Database Performance Engine",
  description:
    "Autonomous AIOps dashboard for real-time database performance monitoring, anomaly detection, and query optimization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-scroll-behavior="smooth">
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
