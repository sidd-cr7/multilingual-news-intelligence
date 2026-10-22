import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "Multilingual News Intelligence", description: "Explainable News Framing & Bias Analysis" };
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
