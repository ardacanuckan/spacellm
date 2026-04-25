import Link from "next/link";
import type { ReactNode } from "react";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/playground", label: "Playground" },
  { href: "/architecture", label: "Architecture" },
  { href: "/roadmap", label: "Roadmap" },
];

export function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b bg-(--color-background)/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link href="/" className="flex items-center gap-2">
            <span className="inline-block size-2 rounded-full bg-(--color-radiation)" />
            <span className="font-mono text-sm tracking-tight">spacellm</span>
            <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-(--color-muted-foreground)">
              v0.3.0-dev
            </span>
          </Link>
          <nav className="flex items-center gap-6 text-sm">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-(--color-muted-foreground) transition-colors hover:text-(--color-foreground)"
              >
                {item.label}
              </Link>
            ))}
            <a
              href="https://github.com/ardacanuckan/spacellm"
              target="_blank"
              rel="noreferrer"
              className="text-(--color-muted-foreground) transition-colors hover:text-(--color-foreground)"
            >
              GitHub
            </a>
          </nav>
        </div>
      </header>
      <main className="flex-1">{children}</main>
      <footer className="border-t">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6 text-xs text-(--color-muted-foreground)">
          <span>© 2026 SpaceLLM Contributors · Apache-2.0</span>
          <span className="font-mono">PyTorch for space.</span>
        </div>
      </footer>
    </div>
  );
}
