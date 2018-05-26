# SQL

- `_` in `LIKE` clauses are like `.` in regexes (match any 1 char)
- can group by multiple columns
- aggregate functions can't be used in `WHERE` clauses, use `HAVING` instead

- `INNER JOIN` keeps rows with keys in both tables
- when joining on column of same name in tables `a` and `b`, instead of `JOIN ... ON a.id = b.id` can use `JOIN ... USING (id)`
- use `CASE` to do if-then logic in queries
- use `INTO` to save results of `SELECT` into new table

    ```sql
    SELECT NAME,
           continent,
           code,
           surface_area,
           CASE
             WHEN surface_area > 2000000 THEN 'large'
             WHEN surface_area > 350000 THEN 'medium'
             ELSE 'small'
           END AS geosize_group
    INTO geosize_grp
    FROM   countries;
    ```

- `LEFT JOIN` keeps rows with keys in left table regardless of whether there are matches in the right table
  - if have multiple matches in right table, will be row for each match in resulting table
  - keeps all values in left table

- `FULL JOIN` keeps all records in left and right tables

- `CROSS JOIN` takes every combination of records in left table with records in right value


- additive joins add columns to left table
  - e.g. left, right, self, cross, full, inner

- semi join: select data in left table if condition on second table is met
- anti join: select data in left table if condition on second table is not met


- can use subqueries in `WHERE` clause, temporary tables in `FROM` clause, `SELECT` clause

    ```sql
    SELECT countries.local_name,
           subquery.lang_num
    FROM   countries,
           (SELECT code,
                   Count(*) AS lang_num
            FROM   languages
            GROUP  BY code) AS subquery
    WHERE  subquery.code = countries.code
    ORDER  BY lang_num DESC
    ```
