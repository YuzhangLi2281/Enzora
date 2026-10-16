import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, isDemo } from "../../shared/api/client";
import { downloadText, serializeCsv } from "../../shared/lib/download";
import { rememberRequest } from "../runs/requestStorage";
import { useApp } from "../../app/store";

type Message = {
  role: "agent" | "user";
  text: string;
  codons?: Record<string, unknown>[];
};
export function ChatPanel({
  jobId,
  onChange,
  phase = 1,
  status,
}: {
  jobId: string | null;
  onChange: () => void;
  phase?: number;
  status?: string;
}) {
  const { lang, sessionId, runRequest, setRunRequest } = useApp();
  const zh = lang === "zh";
  const [open, setOpen] = useState(false),
    [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const [messages, setMessages] = useState<Message[]>([]),
    [input, setInput] = useState("");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [failedMessage, setFailedMessage] = useState("");
  const panel = useRef<HTMLElement>(null),
    composer = useRef<HTMLTextAreaElement>(null),
    end = useRef<HTMLDivElement>(null);
  const toggle = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);
  const version = useRef(0);
  const drag = useRef<{
    x: number;
    y: number;
    left: number;
    top: number;
  } | null>(null);
  useEffect(() => {
    version.current++;
    setMessages([]);
    setError("");
    setBusy(false);
    setFailedMessage("");
  }, [jobId]);
  useLayoutEffect(() => {
    const clamp = () =>
      setPosition((p) =>
        p
          ? {
              x: Math.max(
                8,
                Math.min(innerWidth - panel.current!.offsetWidth - 8, p.x),
              ),
              y: Math.max(
                72,
                Math.min(innerHeight - panel.current!.offsetHeight - 8, p.y),
              ),
            }
          : p,
      );
    clamp();
    window.addEventListener("resize", clamp);
    return () => window.removeEventListener("resize", clamp);
  }, [open]);
  useEffect(() => {
    if (open) composer.current?.focus();
    else if (wasOpen.current) toggle.current?.focus();
    wasOpen.current = open;
  }, [open]);
  useEffect(() => {
    end.current?.scrollIntoView({
      block: "nearest",
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  }, [messages, busy]);
  const send = async (value = input, retry = false) => {
    const text = value.trim();
    if (!text || busy) return;
    const token = version.current;
    setInput("");
    if (composer.current) composer.current.style.height = "auto";
    setBusy(true);
    setError("");
    setFailedMessage("");
    if (!retry) setMessages((m) => [...m, { role: "user", text }]);
    try {
      const response = await api.chat(text, jobId, sessionId);
      if (token !== version.current) return;
      setMessages((m) => [
        ...m,
        { role: "agent", text: response.message, codons: response.codon_rows },
      ]);
      if (response.tool_result && runRequest && jobId) {
        const updated = {
          ...runRequest,
          selection: {
            ...runRequest.selection,
            weights: response.tool_result.weights,
            confidence: response.tool_result.confidence,
          },
        };
        setRunRequest(updated);
        rememberRequest(jobId, updated);
        onChange();
      }
    } catch (e) {
      if (token === version.current) {
        setError(e instanceof Error ? e.message : String(e));
        setFailedMessage(text);
      }
    } finally {
      if (token === version.current) setBusy(false);
    }
  };
  const suggestions =
    phase === 3 && status === "success"
      ? [
          zh ? "解释 Top-K 的选择依据" : "Explain the Top-K selection",
          zh ? "按大肠杆菌优化密码子" : "Optimize codons for E. coli",
        ]
      : phase === 2
        ? [
            zh ? "解释三个评分器的区别" : "Explain the three scorers",
            zh ? "提高活性的配平权重" : "Give activity more weight",
          ]
        : [
            zh
              ? "如何设置突变数量和阶数？"
              : "How do I set the mutation quota?",
            zh
              ? "PheNom 需要什么底物信息？"
              : "What substrate information does PheNom need?",
          ];
  return (
    <aside
      ref={panel}
      className={`copilot ${open ? "expanded" : ""} ${busy ? "is-busy" : ""}`}
      style={
        position
          ? { left: position.x, top: position.y, right: "auto", bottom: "auto" }
          : undefined
      }
      aria-label="Enzora Assistant"
      onKeyDown={(e) => {
        if (e.key === "Escape" && open) {
          e.stopPropagation();
          setOpen(false);
        }
      }}
    >
      <div
        className="copilot-handle"
        tabIndex={0}
        role="toolbar"
        aria-label={
          zh
            ? "拖动助手，或使用方向键移动"
            : "Drag assistant, or move with arrow keys"
        }
        onKeyDown={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          const shifts: Record<string, [number, number]> = {
            ArrowLeft: [-20, 0],
            ArrowRight: [20, 0],
            ArrowUp: [0, -20],
            ArrowDown: [0, 20],
          };
          const shift = shifts[e.key];
          if (!shift) return;
          e.preventDefault();
          const box = panel.current!.getBoundingClientRect();
          setPosition({
            x: Math.max(
              8,
              Math.min(innerWidth - box.width - 8, box.left + shift[0]),
            ),
            y: Math.max(
              72,
              Math.min(innerHeight - box.height - 8, box.top + shift[1]),
            ),
          });
        }}
        onPointerDown={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          const box = panel.current!.getBoundingClientRect();
          drag.current = {
            x: e.clientX,
            y: e.clientY,
            left: box.left,
            top: box.top,
          };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          setPosition({
            x: Math.max(
              8,
              Math.min(
                innerWidth - panel.current!.offsetWidth - 8,
                d.left + e.clientX - d.x,
              ),
            ),
            y: Math.max(
              72,
              Math.min(
                innerHeight - panel.current!.offsetHeight - 8,
                d.top + e.clientY - d.y,
              ),
            ),
          });
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        <span className="copilot-emblem" aria-hidden="true">
          <svg viewBox="0 0 28 28" fill="none">
            <path
              d="M5 18 11 8l7 12 5-9M5 10l7 10 6-13 5 12"
              stroke="currentColor"
              strokeWidth="1.3"
            />
            <circle cx="11" cy="8" r="2" fill="currentColor" />
            <circle cx="18" cy="20" r="2" fill="currentColor" />
          </svg>
        </span>
        <div className="copilot-title">
          <strong>Enzora Assistant</strong>
          <span>
            {busy
              ? zh
                ? "正在处理"
                : "Working"
              : isDemo()
                ? zh
                  ? "演示助手"
                  : "Demo assistant"
                : zh
                  ? "设计与你协作"
                  : "Your design companion"}
          </span>
        </div>
        <span className="copilot-mode">{isDemo() ? "Demo" : "Gemini"}</span>
        <button
          ref={toggle}
          className="copilot-toggle"
          aria-expanded={open}
          aria-controls="copilot-content"
          aria-label={
            open
              ? zh
                ? "收起助手"
                : "Collapse assistant"
              : zh
                ? "打开助手"
                : "Open assistant"
          }
          onClick={() => setOpen(!open)}
        >
          <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <path
              d={open ? "M5 8l5 5 5-5" : "M5 12l5-5 5 5"}
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      {open && (
        <div id="copilot-content" className="copilot-content">
          <div className="copilot-context">
            <span className="copilot-status-dot" />
            <span>
              {jobId
                ? `${zh ? "当前阶段" : "Current phase"} · ${[zh ? "突变库构建" : "Library construction", zh ? "评分与配平" : "Screening & balance", zh ? "优化与结果" : "Optimization & results"][phase - 1]}`
                : zh
                  ? "开始设计前，也可以先提问"
                  : "Ask a question before starting"}
            </span>
          </div>
          <div
            className="copilot-messages"
            role="log"
            aria-live="polite"
            aria-relevant="additions text"
          >
            {!messages.length && (
              <div className="copilot-welcome">
                <h3>
                  {zh
                    ? "从建库到表达，逐步确认。"
                    : "Build. Balance. Prepare to express."}
                </h3>
                <p>
                  {zh
                    ? "解释 PheNom、EpHod 与 UniStab，调整评分配比，或为选中候选生成宿主偏好的 CDS。"
                    : "Interpret PheNom, EpHod and UniStab, adjust their balance, or prepare host-specific CDS for selected candidates."}
                </p>
                <div className="copilot-suggestions">
                  {suggestions.map((text) => (
                    <button
                      key={text}
                      disabled={busy}
                      onClick={() => void send(text)}
                    >
                      <span>{text}</span>
                      <span aria-hidden="true">↗</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((message, i) => (
              <div key={i} className={`chat-message ${message.role}`}>
                <span className="message-author">
                  {message.role === "user" ? (zh ? "你" : "You") : "Enzora"}
                </span>
                <p>{message.text}</p>
                {message.codons && (
                  <button
                    className="copilot-export"
                    onClick={() =>
                      downloadText(
                        serializeCsv(message.codons!, [
                          "mutation",
                          "scientific_name",
                          "optimized_cds_with_stop",
                          "gc_fraction",
                        ]),
                        "enzora-agent-codons.csv",
                        "text/csv;charset=utf-8",
                      )
                    }
                  >
                    {zh ? "导出优化 CDS" : "Export optimized CDS"} ↗
                  </button>
                )}
              </div>
            ))}
            {busy && (
              <div className="copilot-thinking" role="status">
                <span />
                <span />
                <span />
                <p>
                  {isDemo()
                    ? zh
                      ? "准备演示回复"
                      : "Preparing demo reply"
                    : zh
                      ? "正在调用 Gemini"
                      : "Calling Gemini"}
                </p>
              </div>
            )}
            {error && (
              <div className="copilot-error" role="alert">
                <strong>
                  {zh
                    ? "暂时无法完成这条消息"
                    : "Could not complete this message"}
                </strong>
                <p>{error}</p>
                <button
                  disabled={busy}
                  onClick={() => void send(failedMessage, true)}
                >
                  {zh ? "重试" : "Retry"}
                </button>
                <Link to="/settings">
                  {zh ? "检查连接设置" : "Connection settings"} ↗
                </Link>
              </div>
            )}
            <div ref={end} />
          </div>
          <form
            className="copilot-compose"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <textarea
              ref={composer}
              aria-label={zh ? "发送给 Agent 的消息" : "Message the agent"}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.currentTarget.style.height = "auto";
                e.currentTarget.style.height = `${Math.min(96, e.currentTarget.scrollHeight)}px`;
              }}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder={
                zh
                  ? "询问模型，或描述你的设计目标…"
                  : "Ask about a model, or describe your goal…"
              }
              disabled={busy}
              rows={2}
            />
            <div className="compose-bottom">
              <span>
                {isDemo()
                  ? zh
                    ? "模拟回复 · 未调用 Gemini"
                    : "Simulated replies · No Gemini calls"
                  : zh
                    ? "Enter 发送 · Shift+Enter 换行"
                    : "Enter to send · Shift+Enter for a new line"}
              </span>
              <button
                className="copilot-send"
                disabled={!input.trim() || busy}
                type="submit"
                aria-label={zh ? "发送消息" : "Send message"}
              >
                <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
                  <path
                    d="M10 15V5m-4 4 4-4 4 4"
                    stroke="currentColor"
                    strokeWidth="1.7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
          </form>
        </div>
      )}
    </aside>
  );
}
