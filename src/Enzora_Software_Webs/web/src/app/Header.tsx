import githubIcon from "../../../../../resource/fonts/github.png";
import { NavLink, Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, isDemo, setDemo } from "../shared/api/client";
import { t } from "../shared/lib/i18n";
import { useApp } from "./store";
import { Badge } from "../shared/ui/primitives";

const pages = [
  ["/workbench", "workbench"],
  ["/history", "history"],
  ["/tools", "tools"],
  ["/settings", "settings"],
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
      <header className="layout-header">
        <Link to="/workbench" className="brand" aria-label="Enzora">
          <svg
            className="brand-symbol"
            viewBox="0 0 40 40"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M10 10 30 20 10 30V10ZM10 10 26 5M30 20 34 33M10 30 26 35"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <circle cx="10" cy="10" r="4" fill="currentColor" />
            <circle cx="30" cy="20" r="4" fill="currentColor" />
            <circle cx="10" cy="30" r="4" fill="currentColor" />
            <circle cx="26" cy="5" r="2" fill="currentColor" />
            <circle cx="34" cy="33" r="2" fill="currentColor" />
            <circle cx="26" cy="35" r="2" fill="currentColor" />
          </svg>
          <span>
            Enz<span className="brand-accent">ora</span>
            <small>TONGJI-CHINA · 2026</small>
          </span>
        </Link>
        <nav
          className="main-nav"
          aria-label={zh ? "主导航" : "Main navigation"}
        >
          {pages.map(([path, key]) => (
            <NavLink
              key={path}
              className={({ isActive }) =>
                `nav-link ${isActive ? "active" : ""}`
              }
              to={path}
            >
              {t(`nav.${key}`, lang)}
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
              {pages.map(([path, key]) => (
                <NavLink
                  key={path}
                  to={path}
                  onClick={(event) => {
                    event.currentTarget.closest("details")!.open = false;
                  }}
                >
                  {t(`nav.${key}`, lang)}
                </NavLink>
              ))}
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
              aria-label={zh ? "Atlas 蓝黑配色" : "Atlas dark theme"}
              onClick={() => chooseTheme("deep-space")}
            >
              <span className="theme-swatch atlas-swatch" />
              Atlas
            </button>
            <button
              className={theme === "macaron" ? "selected" : ""}
              aria-pressed={theme === "macaron"}
              aria-label={zh ? "马卡龙配色" : "Macaron theme"}
              onClick={() => chooseTheme("macaron")}
            >
              <span className="theme-swatch macaron-swatch" />
              {zh ? "马卡龙" : "Macaron"}
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
          <a
            className="github-link"
            href="https://github.com/YuzhangLi2281/Enzora.github.io"
            target="_blank"
            rel="noopener noreferrer"
            aria-label={
              zh
                ? "在新标签页打开 Enzora GitHub"
                : "Open Enzora on GitHub in a new tab"
            }
          >
            <img src={githubIcon} alt="" width="30" height="30" />
            <span className="github-popover" aria-hidden="true">
              <strong>Enzora on GitHub</strong>
              <span>
                {zh ? "查看项目与软件页面" : "Explore the project and software"}
              </span>
              <span>{zh ? "打开 GitHub" : "Open GitHub"} ↗</span>
            </span>
          </a>
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
