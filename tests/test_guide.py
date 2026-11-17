"""The guide as the owner meets it: in the chat, with examples that really start."""
import json
from pathlib import Path

import pytest
from test_bot import OWNER, Bot, Clock, FakeApi, last_buttons, message, press, tap

from compute_bot import examples, guide, jobs
from compute_bot.recipes import RECIPES

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def env(tmp_path):
    api = FakeApi()
    store = jobs.Store(tmp_path)
    return Bot(store, api, OWNER, tmp_path / "bot.json", clock=Clock(), system_info=lambda: ""), api, store


def test_every_topic_fits_a_telegram_message_and_uses_only_its_html():
    assert len(guide.TOPICS) >= 8
    for topic in guide.TOPICS:
        assert len(topic.text) < 3600, topic.id
        leftover = guide.strip_tags(topic.text)
        assert "<" not in leftover.replace("&lt;", ""), topic.id


def test_guide_never_mentions_a_parameter_or_command_that_does_not_exist(env):
    bot, api, _ = env
    for topic in guide.TOPICS:
        for key in guide.params_in(topic):
            assert key in RECIPES[topic.recipe].params, (topic.id, key)
        for command in guide.commands_in(topic.text):
            bot.handle_update(message("/" + command))
            assert "Nie znam tej komendy" not in api.last_text(), (topic.id, command)


def test_start_welcomes_with_the_guide_and_an_example(env):
    bot, api, _ = env
    bot.handle_update(message("/start"))
    labels = [b["text"] for b in last_buttons(api)]
    assert any("Przewodnik" in x for x in labels) and any("Wypróbuj" in x for x in labels)


def test_guide_lists_topics_and_opens_one(env):
    bot, api, _ = env
    bot.handle_update(message("/guide"))
    data = [b["callback_data"] for b in last_buttons(api)]
    assert "g:md" in data and "g:wyniki" in data
    tap(bot, "g:md")
    shown = [c for c in api.calls if c[0] in ("sendMessage", "editMessageText")][-1][1]["text"]
    assert "RMSD" in shown
    bot.handle_update(message("/guide qsar"))
    assert "Q²" in api.last_text()


@pytest.mark.parametrize("name", list(examples.EXAMPLES))
def test_every_example_opens_a_valid_card_and_starts(env, name):
    bot, api, store = env
    tap(bot, f"t:{name}")
    press(bot, api, "▶️")
    (spec,) = store.specs()
    example = examples.EXAMPLES[name]
    assert spec.entry == example.recipe
    for rel in example.files():
        assert (store.work(1) / rel).exists()


def test_docs_guide_is_generated_from_the_same_text():
    assert (REPO / "docs/przewodnik.md").read_text() == guide.markdown()


def test_daily_check_announces_a_new_version_once_with_an_install_button(tmp_path):
    class Updater:
        can_roll_back = False

        def __init__(self):
            self.checks, self.installed = 0, []

        def latest(self):
            self.checks += 1
            return "v9.9.9"

        def install(self, tag):
            self.installed.append(tag)
            return False

    api, clock, updater = FakeApi(), Clock(), Updater()
    bot = Bot(jobs.Store(tmp_path), api, OWNER, tmp_path / "bot.json", updater=updater, clock=clock,
              system_info=lambda: "")
    bot.tick()
    bot.tick()
    assert updater.checks == 1 and sum("9.9.9" in t for t in api.texts()) == 1
    clock.now += 25 * 3600
    bot.tick()
    assert updater.checks == 2 and sum("9.9.9" in t for t in api.texts()) == 1  # told once
    tap(bot, json.loads(api.sent("sendMessage")[0][1]["reply_markup"])["inline_keyboard"][0][0]["callback_data"])
    assert updater.installed == ["v9.9.9"] and bot.restart
