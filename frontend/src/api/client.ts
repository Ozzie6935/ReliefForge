const API_BASE_URL = (
  import.meta.env?.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "")

export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

export async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (!isRecord(body)) {
      return `Backend returned HTTP ${response.status}.`
    }
    if (typeof body.detail === "string" && body.detail.trim()) {
      return body.detail
    }
    if (Array.isArray(body.detail)) {
      const messages = body.detail.flatMap((item) => {
        if (!isRecord(item) || typeof item.msg !== "string") return []
        return [item.msg]
      })
      if (messages.length > 0) return messages.join(" ")
    }
  } catch {
    // The body was not JSON.
  }
  return `Backend returned HTTP ${response.status}.`
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(apiUrl(path), init)
  } catch {
    throw new ApiError("Unable to reach the backend.")
  }

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }

  try {
    return (await response.json()) as T
  } catch {
    throw new ApiError("Backend returned an unreadable response.", response.status)
  }
}

export function apiGet<T>(path: string): Promise<T> {
  return apiRequest<T>(path)
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError"
}

export async function apiGetArrayBuffer(path: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  let response: Response
  try {
    response = await fetch(apiUrl(path), { method: "GET", signal })
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new ApiError("Unable to reach the backend.")
  }
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }
  try {
    return await response.arrayBuffer()
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new ApiError("Backend returned an unreadable response.", response.status)
  }
}

export async function apiPostNoContent(path: string, body: unknown): Promise<void> {
  let response: Response
  try {
    response = await fetch(apiUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  } catch {
    throw new ApiError("Unable to reach the backend.")
  }
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }
}

export function apiPostJson<T>(path: string, body: unknown): Promise<T> {
  return apiRequest<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

export function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  return apiRequest<T>(path, {
    method: "POST",
    body: form,
  })
}

export function apiPutJson<T>(path: string, body: unknown): Promise<T> {
  return apiRequest<T>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

export async function apiDelete(path: string): Promise<void> {
  let response: Response
  try {
    response = await fetch(apiUrl(path), { method: "DELETE" })
  } catch {
    throw new ApiError("Unable to reach the backend.")
  }
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }
}
