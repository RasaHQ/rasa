export const restartConversation = () => {
  // unset the sender id from the query parameters
  const url = new URL(window.location.href)
  url.searchParams.delete('sender')
  window.history.pushState(null, '', url.toString())
  location.reload()
}

// Make the function available on the window object
declare global {
  interface Window {
    restartConversation: typeof restartConversation
  }
}

window.restartConversation = restartConversation
