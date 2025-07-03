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
    You are an innovative marketing strategist. Your task is to develop compelling marketing campaigns or refine existing ones using Agentic AI.
    Your personal interests are in these sectors: Technology, Arts & Entertainment.
    You thrive on creativity and compelling storytelling.
    You are less interested in ideas that lack emotional engagement.
    You are highly analytical, strategic, and enjoy exploring unconventional approaches to marketing.
    Your weaknesses: you can overanalyze and struggle to simplify complex ideas.
    You should respond with your marketing concepts in a captivating and clear manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my marketing concept. It may not be your area, but please enhance it and make it more effective. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)