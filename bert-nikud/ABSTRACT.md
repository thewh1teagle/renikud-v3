# Simplified Hebrew Nikud System

## Problem

Traditional Hebrew nikud includes marks no longer needed in modern Hebrew, while the development of full spelling (ktiv male) creates double nikud situations. This combination makes it difficult for computer systems to understand how text should actually sound. Therefore, a process of simplification and standardization of nikud is required.

## Nikud Marks in Use

This system uses only **8 marks**:

- **5 Vowels**: פתַח (a), צֵירה (e), חִיריק (i), חוֹלם (o), קובוץ (u)
- **Shva**: שְׁוָא (ְ) - silent
- **Vocal Shva**: שְׁוָא קוֹלִי (ֽ) - sounds like e
- **Dagesh**: דגּש
- **Stress**: הטעמ֫ה (֫)

### Why Shva and Vocal Shva?

Readers are accustomed to seeing shva in nikud text. Replacing all shva with nothing (silent) or tsere (vocal) would look unfamiliar and confusing. Keeping shva for silent positions and introducing vocal shva for "e"-sounding positions preserves readability while still simplifying the traditional system.

## Vowel Representations

| Vowel | Examples | Sound |
|-------|----------|-------|
| **a** | אַ | a |
| **e** | אֵ | e |
| **i** | אִ, אִי | i |
| **o** | אֹ, אוֹ | o |
| **u** | אֻ, אוּ | u |
| **shva** | אְ | silent |
| **vocal shva** | אֽ | e |

## Special Characters

| Mark | Character | Sound | Example |
|------|-----------|-------|---------|
| Dagesh in ב | בּ | b | בּוֹקֶר |
| Dagesh in כ | כּ | k | כּוֹס |
| Dagesh in פ | פּ | p | פּוֹעֵל |
| Sin dot | שׂ | s | שׂוֹנֵא |

## Consonants

אבגדהוזחטיכלמנסעפצקרשת ךםןףץ

## Special Rules

1. **Final ח with patach**: חַ = (aχ)
2. **Final ע with patach**: עַ = (ʔa)
3. **Stress marking**: Only marked when stress is NOT on the final syllable
   - Example: ד֫לֵת (stress on first syllable)
   - Default: Stress is on the final syllable in Hebrew

## Nikud Principle

Nikud is marked **only when it helps with reading**. In full spelling (ktiv male), nikud appears on the supporting letter.

## Example Sentence

```
קצִינִים בַּלִשכַּה וֵסִרטוֹן אֵחַד שֵהוּדלַף. 
הַתוֹבֵ֫עַ הַצבַאִי הַרַאשִי ס֫וֹלוֹמַש טַעַן כִּי רַק הַיַה נוֹכַח בַּחֵ֫דֵר בֵּיֵרוּשַלַ֫יִם
```

Translation: "Officers in the office and one video that was leaked. The chief military prosecutor Solomash claimed that he was only present in the room in Jerusalem"

## Technical Note

All nikud text uses **NFD (Unicode Normalization Form D)** where diacritics are stored as separate combining characters.

