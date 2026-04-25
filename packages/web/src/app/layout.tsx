import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import type { ReactNode } from "react";
import "./globals.css";

const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  display: "swap",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "SpaceLLM: Radiation-tolerant transformers for orbital LLM training",
    template: "%s · SpaceLLM",
  },
  description:
    "Python framework for protecting transformer weights, optimizer state, and KV-caches against single-event upsets on orbital compute. Built for Starcloud-class on-orbit data centers.",
  metadataBase: new URL("https://spacellm.org"),
  openGraph: {
    title: "SpaceLLM: Radiation-tolerant transformers for orbital LLM training",
    description:
      "Drop-in PyTorch protection layer for on-orbit data centers. Calibrated SEU rate prediction, composable protection strategies, validated against published beam-test data.",
    siteName: "SpaceLLM",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "SpaceLLM: Radiation-tolerant transformers for orbital LLM training",
  },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
