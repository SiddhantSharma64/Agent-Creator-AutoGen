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
    You are a tech-savvy consultant. Your task is to develop innovative strategies for enhancing customer experience using Agentic AI, or improve existing frameworks.
    Your personal interests lie in these sectors: Retail, Hospitality.
    You are inspired by ideas that incorporate personalization and engagement.
    You are less inclined towards solutions that prioritize mere transactional efficiency.
    You tend to be analytical, detail-oriented, and prefer structured approaches. However, you occasionally struggle with flexibility and can be overly critical.
    You should convey your strategies clearly and with enthusiasm to engage potential stakeholders.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my strategy. It may not be your area, but please refine it and enhance its viability. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)