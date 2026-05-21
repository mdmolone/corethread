import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"

describe("Theme (Phase 10 / Plan 10-05 / D-27 / THM-01)", () => {
  it("frontend/src/index.css contains a (prefers-color-scheme: dark) block", () => {
    const css = readFileSync(resolve(__dirname, "../index.css"), "utf-8")
    expect(css).toMatch(/@media\s*\(prefers-color-scheme:\s*dark\)/)
  })

  it("THM-01: theme is config-driven without ThemeProvider or toggleTheme wiring", () => {
    const app = readFileSync(resolve(__dirname, "../App.tsx"), "utf-8")
    expect(app).not.toMatch(/ThemeProvider/i)
    expect(app).not.toMatch(/toggleTheme/i)
    expect(app).toMatch(/root\.removeAttribute\("data-theme"\)/)
    expect(app).toMatch(/root\.setAttribute\("data-theme",\s*themeMode\)/)
  })

  it("THM-01: index.css includes config-selectable light and dark theme overrides", () => {
    const css = readFileSync(resolve(__dirname, "../index.css"), "utf-8")
    expect(css).toMatch(/:root\[data-theme="light"\]/)
    expect(css).toMatch(/:root\[data-theme="dark"\]/)
  })

  it("THM-01: index.css contains BOTH light-mode :root vars AND dark-mode :root override inside the media block", () => {
    const css = readFileSync(resolve(__dirname, "../index.css"), "utf-8")
    // Two :root blocks expected: one base, one inside @media (prefers-color-scheme: dark).
    const rootMatches = css.match(/:root\s*\{/g) ?? []
    expect(rootMatches.length).toBeGreaterThanOrEqual(2)
    // The dark-mode block must define --background to a darker value than the light-mode default.
    expect(css).toMatch(/--background:\s*0\s+0%\s+3\.9%/)
  })
})
