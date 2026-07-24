library(tidyverse)

natural_results <- fs::dir_ls("results/eval/attractors/", regexp = "*target_natural_scored.csv", recurse = TRUE) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    params = case_when(
      str_detect(file, "2B") ~ "2B",
      TRUE ~ "4B"
    )
  ) %>%
  select(-file)

all_seed_results <- fs::dir_ls("results/eval/attractors-50seeds", regexp = "attractor_scored*", recurse = TRUE) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    params = case_when(
      str_detect(file, "2B") ~ "2B",
      TRUE ~ "4B"
    ),
    source = case_when(
      str_detect(file, "image") ~ "vision",
      TRUE ~ "language"
    )
  ) %>%
  select(-file)

best_seed_results <- fs::dir_ls("results/eval/attractors/", regexp = "*target_wug", recurse = TRUE) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    params = case_when(
      str_detect(file, "2B") ~ "2B",
      TRUE ~ "4B"
    ),
    source = case_when(
      str_detect(file, "syntax") ~ "language",
      TRUE ~ "vision"
    )
  ) %>%
  select(-file)

best_seed_results %>%
  select(params, source, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  group_by(params, source, correctness, attractors) %>%
  summarize(
    acc = mean(correct)
  ) %>% View()

natural_results %>%
  select(params, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  group_by(params, correctness, attractors) %>%
  summarize(
    acc = mean(correct)
  ) %>% View()

best_seed_results %>%
  select(params, source, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  filter(correctness != "is_correct_all") %>%
  group_by(params, source, attractors) %>%
  summarize(
    acc = mean(correct)
  ) %>%
  ggplot(aes(attractors, acc, color = source)) +
  geom_point() +
  geom_line() + 
  facet_wrap(~params) +
  scale_y_continuous(limits = c(0.5, 1))

natural_acc <- natural_results %>%
  select(params, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  filter(correctness != "is_correct_all") %>%
  group_by(params, attractors) %>%
  summarize(
    acc = mean(correct),
    .groups = "drop"
  ) 
# 
# natural_acc_source <- bind_rows(
#   natural_acc %>% mutate(source = "language"),
#   natural_acc %>% mutate(source = "vision")
# )

all_seed_acc <- all_seed_results %>%
  select(params, source, seed, idx, attractors, is_correct_singular, is_correct_plural, is_correct_all) %>%
  pivot_longer(is_correct_singular:is_correct_all, names_to = "correctness", values_to = "correct") %>%
  filter(correctness != "is_correct_all") %>%
  group_by(params, source, attractors, seed) %>%
  summarize(
    acc = mean(correct)
  ) %>%
  ungroup() %>%
  group_by(params, source, attractors) %>%
  summarize(
    n = n(),
    sd = sd(acc),
    conf = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    acc = mean(acc),
    .groups = "drop"
  )

bind_rows(
  natural_acc %>% mutate(source = "language", type = "natural"),
  natural_acc %>% mutate(source = "vision", type = "natural"),
  all_seed_acc %>% mutate(type = "novel word")
) %>%
  ggplot(aes(attractors, acc, shape = type, linetype = type, color = source, fill = source)) +
  geom_point() +
  geom_line() + 
  geom_ribbon(aes(ymin = acc-conf, ymax = acc+conf), color = NA, alpha = 0.4) +
  facet_wrap(~params) +
  scale_y_continuous(limits = c(0.5, 1))





