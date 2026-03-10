/**
 * Error thrown when socket connection times out
 */
export class SocketTimeoutError extends Error {
  constructor(message = "Socket connection timeout") {
    super(message);
    this.name = "SocketTimeoutError";
  }
}

/**
 * Error thrown when socket connection is unavailable
 */
export class SocketUnavailableError extends Error {
  constructor(message = "Socket connection unavailable") {
    super(message);
    this.name = "SocketUnavailableError";
  }
}
