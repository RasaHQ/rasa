import type { ZodType } from "zod";
import { type BotData, BotDataSchema } from "./types";

async function fetchWithJsonBody(
  url: string,
  options: RequestInit = {},
  body?: unknown,
  segmentUserId?: string,
): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "X-User-Id": segmentUserId || "",
        ...(options.headers || {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    let message = "Network error: Unable to reach the server.";
    if (err instanceof Error && err.message) {
      message += `\nDetails: ${err.message}`;
    }
    throw new Error(message);
  }

  if (!response.ok) {
    const errorData: unknown = await response.json().catch(() => ({}));
    let errorMsg = "An unknown API error occurred.";
    if (
      errorData &&
      typeof errorData === "object" &&
      "error" in errorData &&
      typeof (errorData as { error?: string }).error === "string"
    ) {
      errorMsg = (errorData as { error?: string }).error || errorMsg;
    }
    throw new Error(errorMsg);
  }

  return response;
}

export function validateData<T>(data: unknown, schema: ZodType<T>): T {
  const parsed = schema.safeParse(data);
  if (!parsed.success) {
    throw new Error(
      `Response data validation failed: ${JSON.stringify(parsed.error.issues)}`,
    );
  }
  return parsed.data;
}

const apiRequest = async <TResponse, TRequest = unknown>(
  url: string,
  options: RequestInit = {},
  body?: TRequest,
  responseSchema?: ZodType<TResponse>,
): Promise<TResponse> => {
  const response = await fetchWithJsonBody(url, options, body);
  const data: unknown = await response.json();
  if (responseSchema) {
    return validateData(data, responseSchema);
  }
  return data as TResponse;
}

export const getBotData = async ({
  projectUrl,
  botDataEndpoint,
}: {
  projectUrl: string;
  botDataEndpoint: string;
}): Promise<BotData> => {
  return apiRequest<BotData>(
    `${projectUrl}${botDataEndpoint}`,
    { method: "GET" },
    undefined,
    BotDataSchema,
  );
};


