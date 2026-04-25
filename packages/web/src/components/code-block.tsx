import { cn } from "@/lib/utils";

export function CodeBlock({
  code,
  language,
  className,
}: {
  code: string;
  language?: string;
  className?: string;
}) {
  return (
    <pre
      className={cn(
        "overflow-x-auto rounded-lg border bg-(--color-muted)/60 p-4 font-mono text-[13px] leading-relaxed",
        className,
      )}
    >
      {language ? (
        <span className="mb-2 block text-[10px] uppercase tracking-widest text-(--color-muted-foreground)">
          {language}
        </span>
      ) : null}
      <code>{code}</code>
    </pre>
  );
}
