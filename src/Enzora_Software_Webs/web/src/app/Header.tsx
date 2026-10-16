import { useEffect, useId, useRef, useState } from "react";
import githubIcon from "../../../../../resource/fonts/github.png";
import enzoraLogo from "../../../../../resource/models/enzora-logo.svg";
import { NavLink, Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, isDemo, setDemo } from "../shared/api/client";
import { useApp } from "./store";
import { Badge } from "../shared/ui/primitives";

const pages = [
  ["/workbench", "Workspace"],
  ["/tools", "Model&Tools"],
  ["/teams", "Teams"],
] as const;

export function Header() {
  const { lang, setLang, theme, setTheme } = useApp();
  const [, setParams] = useSearchParams();
  const chooseTheme = (next: "deep-space" | "macaron") => {
    setTheme(next);
    setParams(
      (previous) => {
        const query = new URLSearchParams(previous);
        query.set("theme", next);
        return query;
      },
      { replace: true },
    );
  };
  const demo = isDemo();
  const zh = lang === "zh";
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const sources = useRef<HTMLDivElement>(null);
  const sourcesButton = useRef<HTMLButtonElement>(null);
  const menuId = useId();
  const menuStartAtEnd = useRef(false);
  const focusMenuItem = (last = false) => window.requestAnimationFrame(() => {
    const items = sources.current?.querySelectorAll<HTMLElement>('[role="menuitem"]');
    if (items?.length) items[last ? items.length - 1 : 0].focus();
  });
  useEffect(() => {
    if (!sourcesOpen) return;
    focusMenuItem(menuStartAtEnd.current);
    const outside = (event: Event) => {
      if (event.target instanceof Node && !sources.current?.contains(event.target)) setSourcesOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("focusin", outside);
    };
  }, [sourcesOpen]);

  const query = useQuery({
    queryKey: ["tools"],
    queryFn: () => api.tools(),
    enabled: !demo,
    refetchInterval: 30_000,
  });
  const total = query.data?.length ?? 0;
  const healthy =
    query.data?.filter((tool) => tool.health.available).length ?? 0;
  const status = demo
    ? zh
      ? "演示模式"
      : "Demo mode"
    : query.isError
      ? zh
        ? "后端未连接"
        : "Server offline"
      : query.isPending
        ? zh
          ? "正在连接"
          : "Connecting"
        : `${healthy}/${total} ${zh ? "工具可用" : "tools ready"}`;
  return (
    <>
      <a className="skip-link" href="#main-content">
        {zh ? "跳至主要内容" : "Skip to content"}
      </a>
      <header className="layout-header app-header">
        <Link to="/workbench" className="brand" aria-label="Enzora">
          <img className="brand-symbol" src={enzoraLogo} alt="" width="40" height="40" />
          <span>
            Enz<span className="brand-accent">ora</span>
            <small>TONGJI-CHINA · 2026</small>
          </span>
        </Link>
        <nav
          className="main-nav"
          aria-label={zh ? "主导航" : "Main navigation"}
        >
          {pages.map(([path, label]) => (
            <NavLink
              key={path}
              className={({ isActive }) =>
                `nav-link ${isActive ? "active" : ""}`
              }
              to={path}
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="header-actions">
          <details className="mobile-navigation">
            <summary aria-label={zh ? "页面菜单" : "Page menu"}>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M5 7h14M5 12h14M5 17h14"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </svg>
            </summary>
            <nav aria-label={zh ? "移动端导航" : "Mobile navigation"}>
              {pages.map(([path, label]) => (
                <NavLink
                  key={path}
                  to={path}
                  onClick={(event) => {
                    event.currentTarget.closest("details")!.open = false;
                  }}
                >
                  {label}
                </NavLink>
              ))}
              <NavLink to="/history" onClick={(event) => { event.currentTarget.closest("details")!.open = false; }}>
                {zh ? "运行记录" : "History"}
              </NavLink>
              <NavLink to="/settings" onClick={(event) => { event.currentTarget.closest("details")!.open = false; }}>
                {zh ? "设置" : "Settings"}
              </NavLink>
              <button aria-pressed={demo} onClick={() => setDemo(!demo)}>
                {demo
                  ? zh
                    ? "切换真实模式"
                    : "Use live mode"
                  : zh
                    ? "进入演示"
                    : "Use demo mode"}
              </button>
            </nav>
          </details>
          <div
            className="theme-switch"
            role="group"
            aria-label={zh ? "配色对比" : "Compare themes"}
          >
            <button
              className={theme === "deep-space" ? "selected" : ""}
              aria-pressed={theme === "deep-space"}
              aria-label={zh ? "深色配色" : "Dark theme"}
              onClick={() => chooseTheme("deep-space")}
            >
              <span className="theme-swatch atlas-swatch" />
              Dark
            </button>
            <button
              className={theme === "macaron" ? "selected" : ""}
              aria-pressed={theme === "macaron"}
              aria-label={zh ? "浅色配色" : "Light theme"}
              onClick={() => chooseTheme("macaron")}
            >
              <span className="theme-swatch macaron-swatch" />
              Light
            </button>
          </div>
          <span className="connection-status" role="status">
            <Badge
              kind={
                demo
                  ? "accent"
                  : query.isError
                    ? "warn"
                    : total && healthy === total
                      ? "ok"
                      : "plain"
              }
            >
              {status}
            </Badge>
          </span>
          <button
            className={`ghost small demo-toggle ${demo ? "mode-active" : ""}`}
            aria-pressed={demo}
            onClick={() => setDemo(!demo)}
            title={zh ? "切换演示模式" : "Toggle demo mode"}
          >
            Demo
          </button>
          <button
            className="ghost small"
            onClick={() => setLang(zh ? "en" : "zh")}
            aria-label={zh ? "Switch to English" : "切换为中文"}
          >
            {zh ? "EN" : "中文"}
          </button>
          <div className="sources-dropdown" ref={sources}>
            <button
              ref={sourcesButton}
              className="github-link sources-trigger"
              type="button"
              aria-label={zh ? "项目 GitHub 仓库" : "Project GitHub repositories"}
              aria-haspopup="menu"
              aria-expanded={sourcesOpen}
              aria-controls={menuId}
              onClick={() => { menuStartAtEnd.current = false; setSourcesOpen((open) => !open); }}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                  event.preventDefault();
                  menuStartAtEnd.current = event.key === "ArrowUp";
                  if (sourcesOpen) focusMenuItem(menuStartAtEnd.current);
                  else setSourcesOpen(true);
                }
                if (event.key === "Escape") setSourcesOpen(false);
              }}
            >
              <img src={githubIcon} alt="" width="30" height="30" />
            </button>
            {sourcesOpen && (
              <div
                id={menuId}
                className="sources-menu"
                role="menu"
                aria-label={zh ? "项目仓库与资源" : "Project repositories and resources"}
                onKeyDown={(event) => {
                  const items = [...event.currentTarget.querySelectorAll<HTMLElement>('[role="menuitem"]')];
                  const current = items.indexOf(document.activeElement as HTMLElement);
                  if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
                    event.preventDefault();
                    const index = event.key === "Home" ? 0 : event.key === "End" ? items.length - 1
                      : (current + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
                    items[index]?.focus();
                  }
                  if (event.key === "Escape") {
                    event.preventDefault();
                    setSourcesOpen(false);
                    sourcesButton.current?.focus();
                  }
                }}
              >
                <a role="menuitem" tabIndex={-1} href="https://github.com/YuzhangLi2281/Enzora" target="_blank" rel="noopener noreferrer" onClick={() => setSourcesOpen(false)}>
                  <strong>Enzora <span aria-hidden="true">↗</span></strong>
                  <span>{zh ? "酶设计工作流与 Agent" : "Enzyme-design workflow and Agent"}</span>
                </a>
                <a role="menuitem" tabIndex={-1} href="https://github.com/YuzhangLi2281/Phenom.git" target="_blank" rel="noopener noreferrer" onClick={() => setSourcesOpen(false)}>
                  <strong>Phenom <span aria-hidden="true">↗</span></strong>
                  <span>{zh ? "酶活性预测模型" : "Enzyme-activity prediction model"}</span>
                </a>
                <Link role="menuitem" tabIndex={-1} to="/teams" onClick={() => setSourcesOpen(false)}>
                  <strong>{zh ? "团队与 Wiki 资源" : "Team and Wiki resources"}</strong>
                  <span>{zh ? "项目入口与发布状态" : "Project links and publication status"}</span>
                </Link>
              </div>
            )}
          </div>
        </div>
      </header>
      {!demo && query.isError && (
        <div className="environment-banner offline-banner">
          <span>
            {zh
              ? "暂未连接后端。你仍可使用演示模式预览完整界面。"
              : "The server is unavailable. Explore the interface in demo mode."}
          </span>
          <button className="ghost small" onClick={() => void query.refetch()}>
            {zh ? "重试连接" : "Reconnect"}
          </button>
          <button className="small" onClick={() => setDemo(true)}>
            {zh ? "进入演示" : "Try demo"}
          </button>
        </div>
      )}
    </>
  );
}
