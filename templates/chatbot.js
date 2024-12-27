document.addEventListener('DOMContentLoaded', function () {
    const geminiAPIKey = 'AIzaSyDhiQ6NBSbzNP4dEWMKyzaE97oVdeASbO0'; // Replace with your key
    const gemini = new GoogleGenerativeAI(geminiAPIKey);
    const model = gemini.getGenerativeModel({ model: 'gemini-1.5-flash' });
  
    const messages = [];
    let userQuestion = '';
    let aiResponse = '';
  
    const chatWidget = document.getElementById('chat-widget');
    const chatBody = document.getElementById('chat-body');
    const userInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-btn');
    const openChatButton = document.getElementById('open-chat-btn');
    const closeChatButton = document.getElementById('close-chat-btn');
    const newChatButton = document.getElementById('new-chat-btn');
  
    sendButton.addEventListener('click', handleSend);
    userInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        handleSend();
      }
    });
    closeChatButton.addEventListener('click', () => chatWidget.style.display = 'none');
    openChatButton.addEventListener('click', () => chatWidget.style.display = 'block');
    newChatButton.addEventListener('click', () => {
      messages.length = 0;
      chatBody.innerHTML = '';
    });
  
    async function geminiResponse(userQuestion) {
      try {
        const prompt = `You are a highly knowledgeable assistant for the Star Light Tracker project. A user has asked the following question about astronomy or star tracking: "${userQuestion}". Provide a clear, concise, and accurate response.`;
  
        const result = await model.generateContent(prompt);
        aiResponse = result ? result.response.text : 'No response received.';
        addBotMessage(aiResponse);
      } catch (error) {
        console.error('Error with Gemini API:', error);
        aiResponse = 'Sorry, there was an error while fetching the response.';
        addBotMessage(aiResponse);
      }
    }
  
    function handleSend() {
      const userMessage = userInput.value.trim();
      if (userMessage) {
        addUserMessage(userMessage);
        userQuestion = userMessage;
        userInput.value = '';
        geminiResponse(userQuestion);
      }
    }
  
    function addUserMessage(text) {
      const messageDiv = document.createElement('div');
      messageDiv.classList.add('chat-message', 'user');
      messageDiv.textContent = text;
      chatBody.appendChild(messageDiv);
      scrollToBottom();
    }
  
    function addBotMessage(text) {
      const messageDiv = document.createElement('div');
      messageDiv.classList.add('chat-message', 'bot');
      messageDiv.textContent = text;
      chatBody.appendChild(messageDiv);
      scrollToBottom();
    }
  
    function scrollToBottom() {
      chatBody.scrollTop = chatBody.scrollHeight;
    }
  
    openChatButton.style.display = 'block';
  });
  