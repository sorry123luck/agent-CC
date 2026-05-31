/** React Query hooks for page model tree. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { deletePageModel, deleteStateTemplate, fetchPageModelTree } from '../api/client';

export function usePageModelTree() {
  return useQuery({
    queryKey: ['page-model-tree'],
    queryFn: fetchPageModelTree,
    refetchInterval: 10000,
  });
}

export function useDeletePageModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (pageModelId: string) => deletePageModel(pageModelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['page-model-tree'] });
      queryClient.invalidateQueries({ queryKey: ['canvases'] });
    },
    onError: (error) => {
      window.alert(`删除页面模型失败：${error instanceof Error ? error.message : String(error)}`);
    },
  });
}

export function useDeleteStateTemplate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (stateTemplateId: string) => deleteStateTemplate(stateTemplateId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['page-model-tree'] });
      queryClient.invalidateQueries({ queryKey: ['canvases'] });
    },
    onError: (error) => {
      window.alert(`删除页面状态失败：${error instanceof Error ? error.message : String(error)}`);
    },
  });
}
