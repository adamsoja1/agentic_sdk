import asyncio

import pytest

from agentic_framework import Agent, Conversation, Crew, ImageInput
from test_agent import FakeClient, chunk, tool_call_delta


def test_images_serialize_for_both_apis_without_changing_text_messages():
    conversation = Conversation()
    conversation.add_user_message('hello')
    images = [ImageInput('https://example.com/a.png', detail='high'), ImageInput.from_bytes(b'image', 'image/png')]
    conversation.add_user_message('Compare', images=images)
    assert conversation.get_response_input()[0]['content'] == 'hello'
    assert conversation.get_response_input()[1]['content'] == [
        {'type': 'input_text', 'text': 'Compare'},
        *[{'type': 'input_image', 'image_url': image.url, 'detail': image.detail} for image in images],
    ]
    assert conversation.get_messages()[1]['content'][1] == {
        'type': 'image_url', 'image_url': {'url': images[0].url, 'detail': 'high'},
    }
    assert images[1].url == 'data:image/png;base64,aW1hZ2U='


def test_image_only_input_and_latest_message():
    conversation = Conversation()
    conversation.add_user_message('', images=['https://example.com/a.png'])
    assert conversation.get_response_input()[0]['content'][0]['type'] == 'input_image'
    assert len(conversation.get_latest_user_images()) == 1
    conversation.add_user_message('next')
    assert conversation.get_latest_user_images() == []


@pytest.mark.parametrize('url,detail', [('file:///tmp/a.png', 'auto'), ('https://example.com/a.png', 'wrong')])
def test_invalid_image_input(url, detail):
    with pytest.raises(ValueError):
        ImageInput(url, detail=detail)


def test_agent_preserves_images_through_tool_iteration():
    from agentic_framework import BaseTool
    client = FakeClient([
        [chunk(tool_calls=[tool_call_delta(0, 'call_1', 'lookup', '{}')])],
        [chunk(content='A cat')],
    ])
    agent = Agent(name='vision', model='test', client=client,
                  tools=[BaseTool(name='lookup', description='lookup', func=lambda: 'ok')])
    assert asyncio.run(agent.invoke('Describe', images=['https://example.com/a.png'])) == 'A cat'
    for call in client.calls:
        assert call['input'][0]['content'][1]['type'] == 'input_image'


def test_crew_forwards_images_on_delegation():
    entry_client = FakeClient([[chunk(tool_calls=[tool_call_delta(0, 'call_1', 'delegate_to_agent_specialist', '{"task":"describe"}')])]])
    target_client = FakeClient([[chunk(content='A cat')]])
    entry = Agent(name='entry', model='test', client=entry_client)
    target = Agent(name='specialist', model='test', client=target_client)
    crew = Crew(agents=[entry, target], entrypoint_agent=entry)
    assert asyncio.run(crew.get_response('Describe', images=['https://example.com/a.png'])) == 'A cat'
    assert target_client.calls[0]['input'][0]['content'][1]['type'] == 'input_image'


def test_ask_agent_receives_images():
    client = FakeClient([[chunk(content='A cat')]])
    entry = Agent(name='entry', model='test')
    target = Agent(name='specialist', model='test', client=client)
    Crew(agents=[entry, target], entrypoint_agent=entry, only_ask_for_info=True)
    entry.conversation.add_user_message('Describe', images=['https://example.com/a.png'])

    async def ask():
        return [event async for event in entry._ask_agent('specialist', 'Describe')]

    asyncio.run(ask())
    assert client.calls[0]['input'][0]['content'][1]['type'] == 'input_image'
