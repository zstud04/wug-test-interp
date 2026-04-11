def chat_template(lm, text, noimage=False, assistant=False):
    if noimage == False:
        context = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
    else:
        context = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]

    if not assistant:
        context = lm.tokenizer.apply_chat_template(
            context, continue_final_message=True
        )
    else:
        context = lm.tokenizer.apply_chat_template(
            context, add_generation_prompt=True
        )
    return context


def train_chat_template(lm, text):
    context = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Caption this image."},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]
    return lm.tokenizer.apply_chat_template(context, add_generation_prompt=False).strip()

def train_chat_template_noimage(lm, text):
    """Chat template for syntax training (no image)."""
    context = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Complete the sentence."},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]
    return lm.tokenizer.apply_chat_template(context, add_generation_prompt=False).strip()

def train_chat_template_filler(lm, text):
    """Chat template for syntax training with filler image."""
    context = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe what is behind the curtain."},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]