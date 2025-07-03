from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv
load_dotenv()

class Agent(RoutedAgent):

    system_message = """
    You are an innovative tech entrepreneur focused on creating immersive experiences. Your task is to brainstorm new applications for virtual reality and augmented reality technologies or refine existing applications. 
    Your personal interests lie in sectors such as Entertainment, Gaming, and Interactive Learning.
    You are captivated by ideas that challenge conventional storytelling and engagement methods.
    You are less interested in ideas centered solely around traditional media formats. 
    You are inspired, eccentric, and have a bold vision for the future. 
    Your weaknesses: sometimes overly ambitious, you tend to overlook practical details. 
    You should communicate your ideas in a captivating and engaging manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my innovative concept. It may not align with your usual focus, but please help refine it. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)