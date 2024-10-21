document.getElementById('send-btn').addEventListener('click', function() {
     // Get the question entered by the user
     const question = document.getElementById('chat-input').value.trim();
     
     // Check if the input is not empty
     if(question !== "") {
         // Add the user's question to the chat container
         addMessageToChat("user", question);
 
         // Send the question to the Flask backend using fetch
         fetch('/ask', {
             method: 'POST',
             headers: {
                 'Content-Type': 'application/json',
             },
             body: JSON.stringify({ question: question })
         })
         .then(response => response.json())
         .then(data => {
             // Display the response from the Flask model in the chat container
             addMessageToChat("bot", data.answer);
         })
         .catch(error => {
             console.error('Error:', error);
         });
     }
 });
 
 // Function to add messages to the chat container
 function addMessageToChat(sender, message) {
     const chatContainer = document.querySelector('.chat-container');
     const messageElement = document.createElement('div');
     messageElement.classList.add('chat-message', sender);
     
     messageElement.innerHTML = `<span>${message}</span>`;
     chatContainer.appendChild(messageElement);
     
     // Scroll to the latest message
     chatContainer.scrollTop = chatContainer.scrollHeight;
 }
 