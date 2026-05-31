/** React Query hooks for window listing. */

import { useQuery } from '@tanstack/react-query';
import { fetchWindows } from '../api/client';

export function useWindows() {
  return useQuery({
    queryKey: ['windows'],
    queryFn: fetchWindows,
    refetchInterval: 5000,
  });
}
