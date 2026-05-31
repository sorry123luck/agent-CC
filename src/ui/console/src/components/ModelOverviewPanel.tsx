import { AlertTriangle, Boxes, GitBranch, Layers, ListChecks, Route, Shield } from 'lucide-react';
import type { ReactNode } from 'react';
import type { JobStatusResponse, TransitionGraphResponse, VirtualModelDetail } from '../api/types';

interface ModelOverviewPanelProps {
  model: VirtualModelDetail | null;
  transitions?: TransitionGraphResponse | null;
  jobs?: JobStatusResponse[];
}

export default function ModelOverviewPanel({ model, transitions, jobs = [] }: ModelOverviewPanelProps) {
  if (!model) {
    return (
      <div className="flex flex-col h-full items-center justify-center px-4" style={{ background: 'var(--prc-surface-1)' }}>
        <Layers size={18} className="prc-text-dim mb-2" />
        <span className="prc-text-dim text-xs text-center">选择模型控件查看详情</span>
      </div>
    );
  }

  const layerCounts = readLayerCounts(model.model_layers);
  const appShell = model.app_shell ?? {};
  const shellRegions = Array.isArray(appShell.shared_regions) ? appShell.shared_regions.length : 0;
  const candidates = Array.isArray(model.candidates) ? model.candidates : [];
  const regionTemplates = Array.isArray(model.region_templates) ? model.region_templates : [];
  const transitionEdges = Array.isArray(transitions?.transitions) ? transitions.transitions : [];
  const recentJobs = Array.isArray(jobs) ? jobs.slice(0, 5) : [];
  const missing = Array.isArray(model.missing_suggestions) ? model.missing_suggestions : [];
  const visibleItems = Array.isArray(model.visible_items) ? model.visible_items : [];

  return (
    <div className="flex flex-col h-full overflow-y-auto" style={{ background: 'var(--prc-surface-1)' }}>
      <div className="px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--prc-border)' }}>
        <div className="flex items-center gap-1.5">
          <Shield size={12} style={{ color: 'var(--prc-accent)' }} />
          <span className="prc-text text-xs font-semibold truncate">{model.state_label || model.page_class || '页面模型'}</span>
        </div>
        <div className="mt-1 prc-text-dim mono" style={{ fontSize: '0.56rem' }}>
          {(model.state_template_id || '').slice(0, 8)} · {candidates.length} candidates
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3">
        <Section title="模型分层" icon={<Boxes size={10} />}>
          <Metric label="固定控件" value={String(layerCounts.fixed)} />
          <Metric label="动态内容" value={String(layerCounts.dynamic)} />
          <Metric label="结构区域" value={String(layerCounts.container)} />
          <Metric label="人工覆盖" value={String(layerCounts.override)} />
        </Section>

        <Section title="应用外壳" icon={<Layers size={10} />}>
          <Metric label="应用" value={String(appShell.app_id ?? model.app_id ?? '-')} />
          <Metric label="页面类" value={model.page_class || '-'} />
          <Metric label="共享区域" value={String(shellRegions)} />
        </Section>

        {regionTemplates.length > 0 && (
          <Section title="区域模板" icon={<ListChecks size={10} />}>
            {regionTemplates.slice(0, 8).map((region, index) => (
              <Metric
                key={`${String(region.region_id ?? index)}`}
                label={String(region.role ?? region.region_id ?? `区域 ${index + 1}`)}
                value={`${String(region.element_count ?? 0)} 项`}
              />
            ))}
          </Section>
        )}

        {visibleItems.length > 0 && (
          <Section title="可见动态项" icon={<ListChecks size={10} />}>
            {visibleItems.slice(0, 8).map((item, index) => (
              <Metric
                key={`${String(item.item_id ?? index)}`}
                label={String(item.text ?? item.item_type ?? `项目 ${index + 1}`)}
                value={String(item.actionability ?? item.confidence ?? '-')}
              />
            ))}
          </Section>
        )}

        {missing.length > 0 && (
          <Section title="遗漏建议" icon={<AlertTriangle size={10} />}>
            {missing.slice(0, 6).map((item, index) => (
              <div key={`${String(item.suggestion_id ?? index)}`} className="space-y-0.5">
                <div className="prc-text" style={{ fontSize: '0.62rem', lineHeight: 1.35 }}>
                  {String(item.description ?? item.type ?? `建议 ${index + 1}`)}
                </div>
                <div className="prc-text-dim mono" style={{ fontSize: '0.52rem' }}>
                  {String(item.recommended_next_step ?? 'review')} · {String(item.confidence ?? '-')}
                </div>
              </div>
            ))}
          </Section>
        )}

        {transitions && (
          <Section title="状态跳转" icon={<Route size={10} />}>
            {transitionEdges.length === 0 ? (
              <Metric label="跳转记录" value="暂无" />
            ) : transitionEdges.slice(0, 6).map((edge, index) => (
              <div key={`${edge.from_page_class}-${edge.to_page_class}-${index}`} className="space-y-0.5">
                <div className="flex items-center gap-1 prc-text" style={{ fontSize: '0.6rem' }}>
                  <span className="truncate">{edge.from_page_class}</span>
                  <GitBranch size={9} className="prc-text-dim flex-shrink-0" />
                  <span className="truncate">{edge.to_page_class}</span>
                </div>
                <div className="prc-text-dim mono" style={{ fontSize: '0.52rem' }}>
                  {edge.trigger_action || 'observe'} · {Math.round(edge.success_rate * 100)}% · {edge.observe_count} 次
                </div>
              </div>
            ))}
          </Section>
        )}

        {recentJobs.length > 0 && (
          <Section title="后台任务" icon={<GitBranch size={10} />}>
            {recentJobs.map((job) => (
              <Metric
                key={job.job_id}
                label={job.job_type}
                value={job.state === 'failed' ? `失败: ${job.error || '-'}` : job.state}
              />
            ))}
          </Section>
        )}
      </div>
    </div>
  );
}

function readLayerCounts(raw: Record<string, unknown>): { fixed: number; dynamic: number; container: number; override: number } {
  const counts = raw?.counts;
  if (counts && typeof counts === 'object') {
    const obj = counts as Record<string, unknown>;
    return {
      fixed: Number(obj.fixed ?? obj.fixed_controls ?? 0),
      dynamic: Number(obj.dynamic ?? obj.dynamic_content ?? 0),
      container: Number(obj.container ?? obj.containers ?? 0),
      override: Number(obj.override ?? obj.overrides ?? 0),
    };
  }
  return { fixed: 0, dynamic: 0, container: 0, override: 0 };
}

function Section({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <div>
      <div className="flex items-center gap-1 mb-1.5">
        <span className="prc-text-dim">{icon}</span>
        <span className="prc-text-dim font-medium" style={{ fontSize: '0.65rem' }}>{title}</span>
      </div>
      <div className="space-y-1 pl-3">{children}</div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-2">
      <span className="prc-text-dim flex-shrink-0" style={{ fontSize: '0.62rem', minWidth: '64px' }}>{label}</span>
      <span className="prc-text" style={{ fontSize: '0.62rem', wordBreak: 'break-word', lineHeight: 1.4 }}>{value}</span>
    </div>
  );
}
