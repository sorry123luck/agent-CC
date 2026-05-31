/** React Query hooks for canvas browsing. */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  fetchCanvases,
  fetchCanvasDetail,
  fetchCanvasProcessing,
  deleteCanvas,
  observeWindow,
  triggerRoiVlmSupplement,
  triggerVLMSemantic,
} from '../api/client';

const terminalProcessingStates = [
  'enhanced_ready',
  'semantic_ready',
  'semantic_timeout',
  'semantic_partial',
  'semantic_failed',
  'semantic_late_merged',
  'semantic_late_failed',
  'failed',
  'cancelled',
  'local_ready',
];

export function useCanvases() {
  return useQuery({
    queryKey: ['canvases'],
    queryFn: fetchCanvases,
    refetchInterval: 5000,
  });
}

export function useCanvasDetail(canvasId: string | null) {
  return useQuery({
    queryKey: ['canvas', canvasId],
    queryFn: () => fetchCanvasDetail(canvasId!),
    enabled: !!canvasId,
    refetchInterval: (query) => {
      const state = query.state.data?.processing_state;
      return state && !terminalProcessingStates.includes(state) ? 1500 : false;
    },
  });
}

export function useCanvasProcessing(canvasId: string | null) {
  return useQuery({
    queryKey: ['canvas-processing', canvasId],
    queryFn: () => fetchCanvasProcessing(canvasId!),
    enabled: !!canvasId,
    refetchInterval: (query) => {
      const state = query.state.data?.processing_state;
      return state && !terminalProcessingStates.includes(state) ? 1500 : false;
    },
  });
}

export function useObserve() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      hwnd,
      allowVlm,
      forceVlm,
      asyncEnhance,
    }: { hwnd: number; allowVlm?: boolean; forceVlm?: boolean; asyncEnhance?: boolean }) =>
      observeWindow(hwnd, allowVlm, forceVlm, asyncEnhance),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['canvases'] });
      queryClient.invalidateQueries({ queryKey: ['page-model-tree'] });
    },
  });
}

export function useDeleteCanvas() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (canvasId: string) => deleteCanvas(canvasId),
    onSuccess: (_data, canvasId) => {
      queryClient.invalidateQueries({ queryKey: ['canvases'] });
      queryClient.invalidateQueries({ queryKey: ['page-model-tree'] });
      queryClient.removeQueries({ queryKey: ['canvas', canvasId] });
      queryClient.removeQueries({ queryKey: ['canvas-processing', canvasId] });
    },
    onError: (error) => {
      window.alert(`删除观察截图失败：${error instanceof Error ? error.message : String(error)}`);
    },
  });
}

export function useVLMSemantic() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      canvasId,
      force,
      sync,
      taskMode,
      candidateIds,
    }: { canvasId: string; force?: boolean; sync?: boolean; taskMode?: string; candidateIds?: string[] }) =>
      triggerVLMSemantic(canvasId, force, sync, taskMode, candidateIds ?? []),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['canvases'] });
      queryClient.invalidateQueries({ queryKey: ['canvas'] });
      queryClient.invalidateQueries({ queryKey: ['canvas-processing'] });
    },
  });
}

export function useRoiVlmSupplement() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      canvasId,
      roiIds,
      deadlineMs,
    }: { canvasId: string; roiIds?: string[]; deadlineMs?: number }) =>
      triggerRoiVlmSupplement(canvasId, roiIds ?? [], deadlineMs ?? 2000),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['canvas', variables.canvasId] });
      queryClient.invalidateQueries({ queryKey: ['canvases'] });
      for (const delay of [1200, 3000, 5500]) {
        window.setTimeout(() => {
          queryClient.invalidateQueries({ queryKey: ['canvas', variables.canvasId] });
        }, delay);
      }
    },
  });
}
