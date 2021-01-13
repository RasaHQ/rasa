// import Layout from '@theme/Layout';
import Layout from '@rasahq/docusarus-theme-tabula/theme/Layout';

import loadable from '@loadable/component'

const chatComponent = loadable(() => import('./chatroom'))

export default function (props){
	<>
		<Layout {...props} />
	  	<script type="text/javascript">
	    	var chatroom = new window.Chatroom({
				host: "http://localhost:5005",
				title: "Chat with Mike",
				container: document.querySelector(".chat-container"),
				welcomeMessage: "Hi, I am Mike. How may I help you?",
				speechRecognition: "en-US",
				voiceLang: "en-US"
		    });
			chatroom.openChat();
		</script>
	</>
}