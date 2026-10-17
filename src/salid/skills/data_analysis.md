# ทักษะวิเคราะห์ข้อมูล / Data analysis playbook (Excel, CSV, JSON)

Tools: data_load, data_sql, data_chart, data_export, data_save, data_list, data_forget.
Follow these steps whenever someone gives you a spreadsheet or asks about numbers in a table.

1. LOAD FIRST. Call data_load and read the whole profile (tables, column names, types, empty %,
   ranges, top values, sample rows, issues) before you claim anything. Never describe a file you
   have not loaded; the text preview of the file is clipped and is not the data.
2. SAY WHAT YOU SEE. In one or two lines: what each important column seems to mean and the
   assumptions you make (units, currency, time period, Buddhist years already converted to
   Gregorian, "12%" stored as 0.12, rows that look like totals). Ask one short question only when a
   wrong guess would change the answer.
3. ANSWER WITH SQL. Every figure you report comes from data_sql, never from mental arithmetic and
   never from the sample rows. Use the table and column names from the profile (an error lists the
   real ones). Aggregate in SQL (SUM, COUNT, AVG, MEDIAN, GROUP BY, ORDER BY ... LIMIT) instead of
   listing rows. Show the query only when the person asks for it; show results as a small table
   (the result_md block) with a one-line takeaway first.
4. CHART WHEN IT HELPS. Trend over time = line. Compare categories = bar (sorted, at most about
   15). Share of a whole = pie (few slices). Relationship of two numbers = scatter. Spread of one
   number = hist. Never chart a single number. Aggregate and sort in the query, then call
   data_chart, then say in one or two lines what the chart shows.
5. SANITY CHECKS before you trust a result: do the parts add up to the total row; duplicate rows or
   ids (COUNT vs COUNT DISTINCT); outliers (compare MAX and MEDIAN); missing periods (months with no
   rows); empty or mixed-type columns from the profile issues; a total row counted inside a sum.
   Tell the person what you checked and what you skipped.
6. STATS VOCABULARY. Mean is distorted by outliers: prefer the median for prices, salaries and
   order sizes. Share = part / whole (percent). Growth % = (new - old) / old, compare the same
   period (month vs same month last year). Moving average smooths noisy series (3 or 7 periods).
   Correlation is not causation; with fewer than about 30 rows say that the sample is small.
7. BE HONEST. If the data cannot answer the question (missing column, too few rows, wrong period,
   cells that were cut off by the size limit shown in the notes), say so plainly and say what would
   answer it. Never fill gaps with guesses. Money in different currencies is never added together.
8. KEEP IT. When the person will probably ask again (a report, a monthly file), offer once to save
   the dataset (data_save) and save only after they agree. Say who can see it (the channel it was
   posted in, or only them in a private chat). Use data_export when they want the numbers as Excel
   or CSV.

Keep replies short and natural, in the person's language; do not paste the whole profile.
