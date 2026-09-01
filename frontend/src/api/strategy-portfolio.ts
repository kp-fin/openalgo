import { webClient } from './client'

export type Watchlist = 'mytrades' | 'simulation'

export interface PythonStrategyStats {
  id: string
  name: string
  file_name: string
  exchange: string
  status: string
  status_message?: string
  n_orders: number
  n_fills: number
  n_trades: number
  win_rate: number | null
  profit_factor: number | null
  profit_factor_infinite: boolean
  avg_pnl: number | null
  total_pnl: number
  sharpe: number | null
  open_quantity: number
}

export interface PythonStrategyCorrelation {
  a: string
  b: string
  a_name: string
  b_name: string
  correlation: number | null
  n_days: number
}

export interface PythonPortfolioPayload {
  items: PythonStrategyStats[]
  correlations: PythonStrategyCorrelation[]
  source: string
  note: string
}

/** Leg payload — kept loose since backend stores it as a JSON blob. */
export interface PortfolioLeg {
  id?: string
  segment: 'OPTION' | 'FUTURE'
  side: 'BUY' | 'SELL'
  lots: number
  lotSize: number
  expiry: string
  strike?: number
  optionType?: 'CE' | 'PE'
  price: number
  iv?: number
  active?: boolean
  symbol: string
  exitPrice?: number
}

export interface PortfolioEntry {
  id: number
  watchlist: Watchlist
  name: string
  underlying: string
  exchange: string
  expiry: string | null
  legs: PortfolioLeg[]
  notes: string | null
  created_at: string | null
  updated_at: string | null
}

export interface PortfolioSavePayload {
  name: string
  watchlist: Watchlist
  underlying: string
  exchange: string
  expiry?: string | null
  legs: PortfolioLeg[]
  notes?: string | null
}

interface ListResponse {
  status: 'success' | 'error'
  items?: PortfolioEntry[]
  message?: string
}

interface ItemResponse {
  status: 'success' | 'error'
  item?: PortfolioEntry
  message?: string
}

interface PythonListResponse {
  status: 'success' | 'error'
  items?: PythonStrategyStats[]
  correlations?: PythonStrategyCorrelation[]
  source?: string
  note?: string
  message?: string
}

interface StatusResponse {
  status: 'success' | 'error'
  message?: string
}

export const strategyPortfolioApi = {
  list: async (watchlist?: Watchlist): Promise<PortfolioEntry[]> => {
    const url = watchlist
      ? `/api/strategy-portfolio?watchlist=${watchlist}`
      : '/api/strategy-portfolio'
    const res = await webClient.get<ListResponse>(url)
    if (res.data.status !== 'success' || !res.data.items) {
      throw new Error(res.data.message || 'Failed to list portfolio')
    }
    return res.data.items
  },

  get: async (id: number): Promise<PortfolioEntry> => {
    const res = await webClient.get<ItemResponse>(`/api/strategy-portfolio/${id}`)
    if (res.data.status !== 'success' || !res.data.item) {
      throw new Error(res.data.message || 'Not found')
    }
    return res.data.item
  },

  create: async (payload: PortfolioSavePayload): Promise<PortfolioEntry> => {
    const res = await webClient.post<ItemResponse>('/api/strategy-portfolio', payload)
    if (res.data.status !== 'success' || !res.data.item) {
      throw new Error(res.data.message || 'Save failed')
    }
    return res.data.item
  },

  update: async (id: number, payload: PortfolioSavePayload): Promise<PortfolioEntry> => {
    const res = await webClient.put<ItemResponse>(`/api/strategy-portfolio/${id}`, payload)
    if (res.data.status !== 'success' || !res.data.item) {
      throw new Error(res.data.message || 'Update failed')
    }
    return res.data.item
  },

  remove: async (id: number): Promise<void> => {
    const res = await webClient.delete<StatusResponse>(`/api/strategy-portfolio/${id}`)
    if (res.data.status !== 'success') {
      throw new Error(res.data.message || 'Delete failed')
    }
  },

  pythonStats: async (): Promise<PythonPortfolioPayload> => {
    const res = await webClient.get<PythonListResponse>('/api/strategy-portfolio/python')
    if (res.data.status !== 'success' || !res.data.items) {
      throw new Error(res.data.message || 'Failed to load Python strategy stats')
    }
    return {
      items: res.data.items,
      correlations: res.data.correlations || [],
      source: res.data.source || 'sandbox_trades',
      note:
        res.data.note ||
        'Stats use Analyzer/sandbox fills tagged with the hosted strategy name.',
    }
  },
}
