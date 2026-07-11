import { formatDistanceToNow } from 'date-fns';

export function formatDate(date: string | Date): string {
  return formatDistanceToNow(new Date(date), { addSuffix: true });
}

export function formatNumber(num: number): string {
  return new Intl.NumberFormat().format(num);
}
