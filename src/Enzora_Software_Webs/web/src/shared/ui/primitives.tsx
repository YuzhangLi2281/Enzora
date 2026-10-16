/** Small shared UI primitives. */

import type { ReactNode } from "react";
import { useApp } from "../../app/store";
import { t } from "../lib/i18n";

export function Panel(props: {
  title: string;
  icon?: string;
  glow?: boolean;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`glass-panel ${props.glow ? "glow" : ""} ${props.className ?? ""}`}
    >
      <header className="panel-head">
        <span className="dot" style={{ background: props.icon ?? undefined }} />
        <h2>{props.title}</h2>
        <span className="spacer" />
        {props.right}
      </header>
      <div className="panel-body">{props.children}</div>
    </section>
  );
}

export function Badge(props: {
  kind?: "ok" | "warn" | "err" | "accent" | "plain";
  mono?: boolean;
  children: ReactNode;
  title?: string;
}) {
  const kind = props.kind ?? "plain";
  return (
    <span
      className={`badge ${kind !== "plain" ? kind : ""} ${props.mono ? "mono" : ""}`}
      title={props.title}
    >
      {props.children}
    </span>
  );
}

export function Chips(props: {
  options: { key: string; label: string }[];
  value: string | null;
  onChange: (key: string) => void;
}) {
  return (
    <div className="chip-row">
      {props.options.map((opt) => (
        <button
          key={opt.key}
          className={`chip ${props.value === opt.key ? "active" : ""}`}
          onClick={() => props.onChange(opt.key)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function Stat(props: {
  label: string;
  value: ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="muted-box" style={{ padding: "6px 10px" }}>
      <div className="small dim">{props.label}</div>
      <div className={props.mono ? "mono" : ""} style={{ fontWeight: 700 }}>
        {props.value}
      </div>
    </div>
  );
}

export function Empty(props: { text: string }) {
  return (
    <div className="empty-state">
      <span className="empty-orbit" aria-hidden="true">
        ∷
      </span>
      <p>{props.text}</p>
    </div>
  );
}

export function Loading() {
  const lang = useApp((s) => s.lang);
  return <Empty text={`${t("common.loading", lang)}`} />;
}

export function ErrorNote(props: { text: string }) {
  return (
    <div className="error-text small muted-box" role="alert">
      {props.text}
    </div>
  );
}

export function fmt(n: unknown, digits = 3): string {
  if (n === null || n === undefined || n === "") return "—";
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}
