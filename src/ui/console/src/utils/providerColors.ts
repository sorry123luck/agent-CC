/** 感知源颜色映射（统一使用 CSS 变量）。 */
export const providerColor: Record<string, string> = {
  uia: 'var(--prc-accent)',
  ocr: 'var(--prc-warning)',
  vision: '#a78bfa',
  vlm: '#22d3ee',
  dom: 'var(--prc-confirmed)',
  memory: 'var(--prc-muted)',
  omniparser: '#a78bfa',
  heuristic: 'var(--prc-muted)',
};
