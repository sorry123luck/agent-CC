import { Component, type ErrorInfo, type ReactNode } from 'react';

interface AppErrorBoundaryProps {
  children: ReactNode;
}

interface AppErrorBoundaryState {
  error: Error | null;
}

export default class AppErrorBoundary extends Component<AppErrorBoundaryProps, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Console render failed:', error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--prc-bg, #0e1117)',
            color: 'var(--prc-text, #e2e8f0)',
            padding: 24,
          }}
        >
          <div
            style={{
              width: 'min(560px, 100%)',
              border: '1px solid var(--prc-border, #2d3748)',
              borderRadius: 6,
              background: 'var(--prc-surface, #151b26)',
              padding: 16,
            }}
          >
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>界面渲染失败</div>
            <div style={{ fontSize: 12, color: 'var(--prc-text-dim, #94a3b8)', marginBottom: 12 }}>
              前端遇到了异常。请刷新页面；错误详情已写入浏览器控制台。
            </div>
            <pre
              style={{
                maxHeight: 180,
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontSize: 11,
                color: '#fca5a5',
                background: 'rgba(239,68,68,0.08)',
                border: '1px solid rgba(239,68,68,0.22)',
                borderRadius: 4,
                padding: 8,
              }}
            >
              {this.state.error.message}
            </pre>
            <button
              className="prc-btn"
              style={{ marginTop: 12, height: 28, padding: '0 12px', fontSize: 12 }}
              onClick={() => window.location.reload()}
            >
              刷新页面
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
