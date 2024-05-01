from typing import List
import time
import torch  # Make sure torch is installed
import fire
from llama import Llama, Tokenizer

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 128,
    max_batch_size: int = 4,
):
    # Create the generator and tokenizer
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    tokenizer = Tokenizer(tokenizer_path)

    prompts: List[str] = [
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush giraffe => girafe peluche
        cheese =>""",
    ]
    
    additional_prompts = [
        "The quickest way to double your money is to",
        "What do you think about the future of artificial intelligence in medicine?",
        "Explain the difference between classical and operant conditioning.",
        "In the next decade, space travel will",
        "A comprehensive guide to managing personal finances includes",
        "Describe the lifecycle of a butterfly.",
        "What were the main causes of the World War II?",
        "The recipe for making a perfect cheesecake involves",
        "How does quantum computing differ from traditional computing?",
        "An effective workout routine for building muscle should include",
        "The history of the Roman Empire begins with",
        "Tips for first-time home buyers usually include",
        "Explain the role of photosynthesis in the environment.",
        "The philosophical question of 'What is truth?' can be approached by",
        "Early childhood education impacts development by",
        "A day in the life of a software engineer involves",
        "Strategies for reducing stress during exams include",
        "How to prepare a professional resume for a job application:",
        "The significance of Shakespeare's work in modern literature is",
        "Advancements in renewable energy technologies are",
        "What is the impact of social media on teenage mental health?",
        "The plot of '1984' by George Orwell centers around",
        "Characteristics of Baroque art include",
        "Discuss the evolution of communication technology from the 20th century to today.",
        "What are the benefits and risks of genetically modified foods?",
        "A detailed explanation of the water cycle and its importance to Earth's ecosystems.",
        "How to plan and execute an effective marketing campaign:",
        "Describe the impact of globalization on small businesses.",
        "Innovations in automotive technology have led to",
        "The principles of sustainable design include",
        "A breakdown of the human digestive system and its functions.",
        "Tips for improving your digital photography skills include",
        "The economic theories of John Maynard Keynes explain",
        "A guide to hiking the Appalachian Trail would tell you",
        "What are the ethical implications of cloning technologies?",
        "The structure of DNA was first described by",
        "How to create a successful online business from scratch:",
        "The impact of global warming on polar regions is",
        "Methods for teaching math effectively in elementary schools include",
        "What are the long-term effects of sleep deprivation?",
        "A poem about the sea could start with the line",
        "The role of the United Nations in global peacekeeping is",
        "A brief history of the Internet from its inception to the present day.",
        "How do vaccines work to prevent diseases?",
        "Techniques for writing a compelling novel include",
        "The benefits of yoga for mental and physical health are",
        "A critique of modernist architecture focuses on",
        "How to maintain a healthy work-life balance:",
        "Describe the process of photosynthesis and its significance.",
        "The cultural significance of traditional dances in Africa.",
        "A comparison of Eastern and Western philosophies reveals",
        "Safety measures for online transactions should include",
        "What is the role of artificial intelligence in cybersecurity?",
        "Explain the process of cell division in multicellular organisms.",
        "Strategies for effective conflict resolution in teams include",
        "The best practices for data encryption and privacy protection are",
        "How to train for a marathon:",
        "The influence of ancient Greek philosophy on modern thought is",
        "What are the consequences of deforestation on biodiversity?",
        "An overview of the French Revolution, including its causes and major events.",
        "Techniques for improving memory recall include",
        "The role of antioxidants in human health is",
        "A guide to the national parks of Canada would highlight",
        "How to write a business proposal:",
        "The psychological effects of loneliness and isolation are",
        "What are the principles of effective leadership?",
        "Describe the technology behind electric vehicles.",
        "The impact of the printing press on European society was",
        "Methods for preserving ancient manuscripts include",
        "Explain the theory of relativity and its scientific implications.",
        "A narrative describing a journey through the mountains might begin with",
        "How does the immune system respond to viral infections?",
        "Techniques for sustainable farming and agriculture include",
        "The artistic movements of the 20th century included",
        "A discussion on the ethics of space exploration involves",
        "How to optimize your computer's performance:",
        "The significance of the Silk Road in historical trade routes is",
        "A critique of postmodern literature focuses on",
        "Describe the main features of Gothic architecture.",
        "The process of making traditional Japanese sushi involves",
        "What are the key factors driving globalization?",
        "An analysis of Beethoven's influence on classical music.",
        "How to effectively manage team projects using digital tools:",
        "The science behind the formation of rainbows is",
        "A guide to meditation techniques for beginners would include",
        "The impact of tourism on small island economies can be significant.",
        "How to start an urban gardening project:",
        "The principles of cognitive-behavioral therapy and its applications."
    ]

    prompts.extend(additional_prompts)

    # Encode prompts to count input tokens (including BOS and EOS tokens)
    input_tokens = sum(len(tokenizer.encode(prompt, bos=True, eos=True)) for prompt in prompts)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # convert to MB

    start_time = time.time()

    #results = generator.text_completion(
    #    prompts,
    #    max_gen_len=max_gen_len,
    #    temperature=temperature,
    #    top_p=top_p,
    #)
    def process_batches(prompts, batch_size):
        start_index = 0
        results = []
        while start_index < len(prompts):
            end_index = start_index + batch_size
            batch_prompts = prompts[start_index:end_index]
            batch_results = generator.text_completion(
                batch_prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            results.extend(batch_results)
            start_index += batch_size
        return results
    results = process_batches(prompts, max_batch_size)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        final_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # convert to MB

    # Count output tokens using the tokenizer (including BOS and EOS tokens)
    output_tokens = sum(len(tokenizer.encode(result['generation'], bos=True, eos=True)) for result in results)
    total_tokens = input_tokens + output_tokens

    # Compute tokens per second for output tokens only and for total tokens
    output_tokens_per_second = output_tokens / elapsed_time
    total_tokens_per_second = total_tokens / elapsed_time

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    print(f"Generated {output_tokens} output tokens in {elapsed_time:.2f} seconds.")
    print(f"Inference speed (output tokens only): {output_tokens_per_second:.2f} tokens/second")
    print(f"Inference speed (input + output tokens): {total_tokens_per_second:.2f} tokens/second")

    if torch.cuda.is_available():
        print(f"Initial GPU Memory: {initial_mem:.2f} MB")
        print(f"Final GPU Memory: {final_mem:.2f} MB")
        print(f"Memory Used During Inference: {final_mem - initial_mem:.2f} MB")

if __name__ == "__main__":
    fire.Fire(main)

