/** React Query hook for canvas querying. */

import { useMutation } from '@tanstack/react-query';
import { queryCanvas } from '../api/client';
import type { QueryTargetModel } from '../api/types';

export function useCanvasQuery() {
  return useMutation({
    mutationFn: ({ canvasId, target, maxResults }: {
      canvasId: string;
      target: QueryTargetModel;
      maxResults?: number;
    }) => queryCanvas(canvasId, target, maxResults),
  });
}
