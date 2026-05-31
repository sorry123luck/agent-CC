import { useEffect, useMemo, useState } from 'react';
import { CheckCircle, Loader2, RefreshCw, Save, Sparkles, X, XCircle } from 'lucide-react';
import {
  fetchSemanticModelerModels,
  fetchSemanticModelerSettings,
  saveSemanticModelerSettings,
  testSemanticModelerSettings,
} from '../api/client';
import type {
  SemanticModelerModelInfo,
  SemanticModelerSettings,
  SemanticModelerSettingsUpdate,
  SemanticModelerTestResponse,
} from '../api/types';

interface VLMSettingsPanelProps {
  open: boolean;
  onClose: () => void;
}

/* ── Provider 预设（快速填充，不绑定 UI 逻辑） ── */

const PROVIDER_PRESETS: Record<string, { endpoint: string; variant: string; model?: string }> = {
  openrouter:  { endpoint: 'https://openrouter.ai/api/v1', variant: 'openrouter' },
  moonshot:    { endpoint: 'https://api.moonshot.cn/v1',    variant: 'moonshot',  model: 'moonshot-v1-128k-vision-preview' },
  qwen:        { endpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1', variant: 'qwen' },
  doubao:      { endpoint: 'https://ark.cn-beijing.volces.com/api/v3', variant: 'doubao' },
  openai:      { endpoint: 'https://api.openai.com/v1',    variant: 'openai' },
  anthropic:   { endpoint: 'https://api.anthropic.com/v1',  variant: '' },
  minimax:     { endpoint: 'https://api.minimax.chat/v1',   variant: '' },
};

const PROVIDER_LABELS: Record<string, string> = {
  disabled: '禁用',
  openrouter: 'OpenRouter',
  moonshot: '月之暗面 / Kimi',
  qwen: '通义千问',
  doubao: '豆包 / 火山方舟',
  openai: 'OpenAI 兼容',
  anthropic: 'Anthropic',
  minimax: 'MiniMax / MiMo',
  mock: 'Mock 测试',
};

function toUpdate(s: SemanticModelerSettings): SemanticModelerSettingsUpdate {
  const { api_key_set: _, ...rest } = s;
  return rest;
}

/** 根据 endpoint 自动识别 provider（旧配置兼容） */
function inferProvider(s: SemanticModelerSettings): SemanticModelerSettingsUpdate {
  const u = toUpdate(s);
  if (u.provider !== 'openai') return u;
  const ep = (u.endpoint || '').toLowerCase();
  const v  = (u.provider_variant || '').toLowerCase();
  for (const [key, preset] of Object.entries(PROVIDER_PRESETS)) {
    if (key === 'openai') continue;
    if (v === key || ep.includes(new URL(preset.endpoint).host)) {
      return { ...u, provider: key, provider_variant: preset.variant };
    }
  }
  return u;
}

/* ── Component ── */

export default function VLMSettingsPanel({ open, onClose }: VLMSettingsPanelProps) {
  const [form, setForm] = useState<SemanticModelerSettingsUpdate>({});
  const [apiKey, setApiKey] = useState('');
  const [apiKeySet, setApiKeySet] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [models, setModels] = useState<SemanticModelerModelInfo[]>([]);
  const [modelFilter, setModelFilter] = useState<'all' | 'multimodal' | 'image' | 'text' | 'free'>('all');
  const [testResult, setTestResult] = useState<SemanticModelerTestResponse | null>(null);
  const [modelListMeta, setModelListMeta] = useState<{ total: number; filtered: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const requestBody = useMemo(() => {
    const body: SemanticModelerSettingsUpdate = { ...form };
    if (apiKey) body.api_key = apiKey;
    return body;
  }, [form, apiKey]);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    setModels([]);
    setModelListMeta(null);
    setTestResult(null);
    fetchSemanticModelerSettings()
      .then((s) => { setForm(inferProvider(s)); setApiKeySet(s.api_key_set); setApiKey(''); })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [open]);

  const update = (p: SemanticModelerSettingsUpdate) => setForm((f) => ({ ...f, ...p }));

  const handleProviderChange = (provider: string) => {
    const preset = PROVIDER_PRESETS[provider];
    update({
      provider,
      endpoint: preset?.endpoint ?? '',
      provider_variant: preset?.variant ?? '',
      model: preset?.model ?? '',
      free_model_only: false,
    });
    setModels([]);
    setModelListMeta(null);
    setTestResult(null);
  };

  const handleSave = async () => {
    setSaving(true); setError(null);
    try {
      const saved = await saveSemanticModelerSettings(requestBody);
      setForm(inferProvider(saved));
      setApiKeySet(saved.api_key_set);
      setApiKey('');
      setTestResult(null);
    } catch (e) { setError(String(e)); }
    finally { setSaving(false); }
  };

  const handleTest = async (runCall: boolean) => {
    setTesting(true); setError(null);
    try { setTestResult(await testSemanticModelerSettings({ ...requestBody, run_call: runCall })); }
    catch (e) { setError(String(e)); }
    finally { setTesting(false); }
  };

  const handleLoadModels = async () => {
    setLoadingModels(true); setError(null);
    try {
      const r = await fetchSemanticModelerModels({ ...requestBody, vision_only: false });
      if (!r.success) { setError(r.error || '模型列表获取失败'); setModels([]); setModelListMeta(null); }
      else { setModels(r.models); setModelListMeta({ total: r.total_count, filtered: r.filtered_count }); }
    } catch (e) { setError(String(e)); }
    finally { setLoadingModels(false); }
  };

  const visibleModels = useMemo(() => {
    if (modelFilter === 'all') return models;
    if (modelFilter === 'free') return models.filter((m) => m.is_free);
    return models.filter((m) => m.model_type === modelFilter);
  }, [models, modelFilter]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(2,6,23,0.62)' }}>
      <div className="prc-panel flex flex-col" style={{ width: 'min(760px, calc(100vw - 40px))', maxHeight: 'calc(100vh - 48px)', borderColor: 'var(--prc-border)', boxShadow: '0 24px 80px rgba(0,0,0,0.42)' }}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--prc-border)' }}>
          <div className="flex items-center gap-2">
            <Sparkles size={15} style={{ color: '#38bdf8' }} />
            <div>
              <div className="prc-text text-sm font-semibold">VLM 语义模型配置</div>
              <div style={{ fontSize: '0.68rem', color: '#a8b3c7' }}>保存到本机运行时配置，API key 不会写入源码</div>
            </div>
          </div>
          <button className="prc-btn prc-btn-ghost h-7 w-7" onClick={onClose} title="关闭"><X size={14} /></button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {loading ? (
            <div className="prc-text-dim text-xs flex items-center gap-2"><Loader2 size={13} className="animate-spin" /> 正在读取配置...</div>
          ) : (
            <>
              {/* ── 连接配置 ── */}
              <Section title="连接配置">
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Provider">
                    <select className="prc-input h-8" value={form.provider ?? 'disabled'} onChange={(e) => handleProviderChange(e.target.value)}>
                      {Object.entries(PROVIDER_LABELS).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                    </select>
                  </Field>
                  <Field label="模型">
                    <div className="flex gap-1">
                      <input className="prc-input h-8 flex-1" value={form.model ?? ''} onChange={(e) => update({ model: e.target.value })} placeholder="手动输入或从列表选择" list="vlm-model-list" />
                      <button type="button" className="prc-btn prc-btn-ghost h-8 px-2" onClick={handleLoadModels} disabled={loadingModels} title="获取模型列表">
                        {loadingModels ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                      </button>
                    </div>
                    <datalist id="vlm-model-list">
                      {models.map((m) => <option key={m.id} value={m.id}>{m.capability_label}</option>)}
                    </datalist>
                  </Field>
                  <Field label="Endpoint" full>
                    <input className="prc-input h-8 mono" value={form.endpoint ?? ''} onChange={(e) => update({ endpoint: e.target.value })} placeholder="https://api.example.com/v1" />
                  </Field>
                  <Field label="代理 URL">
                    <input className="prc-input h-8 mono" value={form.proxy_url ?? ''} onChange={(e) => update({ proxy_url: e.target.value })} placeholder="留空直连" />
                  </Field>
                  <NumberField label="代理端口" value={form.proxy_port ?? 0} onChange={(v) => update({ proxy_port: v })} />
                  <Field label={<>API Key {apiKeySet && !apiKey ? <span style={{ color: '#64748b' }}>（已保存，留空不变）</span> : ''}</>} full>
                    <input className="prc-input h-8 mono" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder={apiKeySet ? '已保存，输入新 key 可覆盖' : 'sk-...'} />
                  </Field>
                </div>
                <div className="prc-text-dim" style={{ fontSize: '0.6rem', marginTop: 4 }}>
                  代理默认关闭并忽略系统代理；填代理 URL 或端口后仅 VLM API 请求走代理。端口会按 http://127.0.0.1:端口 使用。
                </div>
              </Section>

              {/* ── 推理与性能 ── */}
              <Section title="推理与性能">
                <div className="grid grid-cols-4 gap-3">
                  <Field label="思考模式">
                    <select className="prc-input h-8" value={form.thinking_mode ?? 'auto'} onChange={(e) => update({ thinking_mode: e.target.value })}>
                      <option value="auto">自动（禁用，快速）</option>
                      <option value="on">启用（深度推理）</option>
                      <option value="off">强制禁用</option>
                    </select>
                  </Field>
                  <Field label="图片最大宽度">
                    <select className="prc-input h-8" value={form.image_max_width ?? 1280} onChange={(e) => update({ image_max_width: Number(e.target.value) })}>
                      <option value={960}>960 px（最快）</option>
                      <option value={1280}>1280 px（推荐）</option>
                      <option value={1600}>1600 px</option>
                      <option value={1920}>1920 px（原始）</option>
                    </select>
                  </Field>
                  <NumberField label="超时秒" value={form.timeout_seconds ?? 60} onChange={(v) => update({ timeout_seconds: v })} />
                  <NumberField label="重试次数" value={form.max_retries ?? 2} onChange={(v) => update({ max_retries: v })} />
                </div>
                <div className="prc-text-dim" style={{ fontSize: '0.6rem', marginTop: 4 }}>
                  思考模式：自动=语义理解不需要深度推理，禁用 thinking 加速响应。图片宽度：1280px 是文字清晰度与 payload 大小的最佳平衡点。
                </div>
              </Section>

              {/* ── 额度与隐私 ── */}
              <Section title="额度与隐私">
                <div className="grid grid-cols-4 gap-3">
                  <NumberField label="每日上限" value={form.daily_call_limit ?? 100} onChange={(v) => update({ daily_call_limit: v })} />
                  <NumberField label="月预算 USD" value={form.monthly_budget_usd ?? 10} onChange={(v) => update({ monthly_budget_usd: v })} />
                  <div className="col-span-2" />
                </div>
                <div className="grid grid-cols-4 gap-3 mt-2">
                  <Toggle label="启用" checked={!!form.enabled} onChange={(v) => update({ enabled: v })} />
                  <Toggle label="仅免费模型" checked={!!form.free_model_only} onChange={(v) => update({ free_model_only: v })} />
                  <Toggle label="脱敏动态内容" checked={form.redact_dynamic_content !== false} onChange={(v) => update({ redact_dynamic_content: v })} />
                  <Toggle label="保存脱敏响应" checked={!!form.save_raw_response} onChange={(v) => update({ save_raw_response: v })} />
                </div>
              </Section>

              {/* ── 操作按钮 ── */}
              <div className="flex items-center gap-2 flex-wrap">
                <button className="prc-btn prc-btn-primary h-8 px-3 flex items-center gap-1.5" onClick={handleSave} disabled={saving}>
                  {saving ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />} 保存
                </button>
                <button className="prc-btn prc-btn-ghost h-8 px-3" onClick={() => handleTest(false)} disabled={testing}>仅检查配置</button>
                <button className="prc-btn prc-btn-ghost h-8 px-3 flex items-center gap-1.5" onClick={handleLoadModels} disabled={loadingModels}>
                  {loadingModels ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />} 获取模型列表
                </button>
                <button className="prc-btn prc-btn-ghost h-8 px-3" onClick={() => handleTest(true)} disabled={testing}>发送测试请求</button>
              </div>

              {/* ── 测试结果 ── */}
              {testResult && <TestResultCard result={testResult} />}
              {error && <div className="rounded border p-2 prc-text text-xs" style={{ borderColor: 'rgba(239,68,68,0.35)', color: 'var(--prc-danger)' }}>{error}</div>}

              {/* ── 模型列表 ── */}
              {models.length > 0 && (
                <>
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1">
                      {([['all', '全部'], ['multimodal', '多模态'], ['image', '图片'], ['text', '文本'], ['free', '免费']] as const).map(([k, l]) => (
                        <button key={k} type="button" className={`prc-btn h-7 px-2 ${modelFilter === k ? 'prc-btn-primary' : 'prc-btn-ghost'}`} onClick={() => setModelFilter(k)}>{l}</button>
                      ))}
                    </div>
                    <div className="prc-text-dim text-xs">{modelListMeta ? `服务端 ${modelListMeta.total} 个，当前 ${visibleModels.length} 个` : `${visibleModels.length} 个`}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto">
                    {visibleModels.map((m) => (
                      <button key={m.id} className="prc-btn prc-btn-ghost text-left p-2 h-auto flex flex-col gap-1"
                        style={{ borderColor: form.model === m.id ? '#38bdf8' : '#334155', background: form.model === m.id ? 'rgba(56,189,248,0.12)' : '#111827', color: '#e5edf7' }}
                        onClick={() => update({ model: m.id })}>
                        <span className="prc-text text-xs">{m.name || m.id}</span>
                        <span className="prc-text-dim mono" style={{ fontSize: '0.58rem' }}>{m.id}</span>
                        <span className="prc-text-dim" style={{ fontSize: '0.56rem' }}>
                          {m.capability_label}{m.is_free ? ' · 免费' : ''}{m.pricing_prompt ? ` · in ${m.pricing_prompt}` : ''}{m.pricing_completion ? ` / out ${m.pricing_completion}` : ''}
                        </span>
                        <span className="prc-text-dim" style={{ fontSize: '0.54rem' }}>
                          输入: {m.input_modalities.length ? m.input_modalities.join(', ') : '未知'} · 输出: {m.output_modalities.length ? m.output_modalities.join(', ') : '未知'}
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Sub-components ── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="prc-text text-xs font-medium mb-2" style={{ color: '#94a3b8' }}>{title}</div>
      {children}
    </div>
  );
}

function Field({ label, full, children }: { label: React.ReactNode; full?: boolean; children: React.ReactNode }) {
  return (
    <label className={`flex flex-col gap-1 ${full ? 'col-span-2' : ''}`}>
      <span style={{ fontSize: '0.65rem', color: '#b6c2d6' }}>{label}</span>
      {children}
    </label>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 prc-text text-xs" style={{ color: '#dbe7f6' }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

function NumberField({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <Field label={label}>
      <input className="prc-input h-8" type="number" value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}

function TestResultCard({ result }: { result: SemanticModelerTestResponse }) {
  return (
    <div className="rounded border p-3 text-xs" style={{ borderColor: result.success ? 'rgba(34,197,94,0.35)' : 'rgba(239,68,68,0.35)', background: 'var(--prc-surface-1)' }}>
      <div className="flex items-center gap-2">
        {result.success ? <CheckCircle size={13} color="var(--prc-confirmed)" /> : <XCircle size={13} color="var(--prc-danger)" />}
        <span className="prc-text">{result.message}</span>
        <span className="prc-text-dim mono">{result.provider}/{result.model}</span>
      </div>
      <div className="mt-2 grid grid-cols-4 gap-2 prc-text-dim">
        <span>耗时 {result.latency_ms}ms</span>
        <span>输入 {result.token_input}</span>
        <span>输出 {result.token_output}</span>
        <span>{result.finish_reason || '-'}</span>
      </div>
      {(result.raw_preview || result.error) && (
        <pre className="mt-2 prc-text-dim mono whitespace-pre-wrap" style={{ fontSize: '0.62rem' }}>{result.raw_preview || result.error}</pre>
      )}
    </div>
  );
}
