# Design Decisions

I made these choices to keep the attendance API consistent, accurate, and reliable.

**Date range:** I treat a trend period of more than 92 days, including both the start and end dates, as invalid. This keeps the limit consistent with a three-month reporting period.

**Rounding:** I return `attendance_pct` to two decimal places, while attendance rates use four decimal places as required by the contract. Moving averages use the reported, already-rounded attendance rates.

**Attendance corrections:** A PATCH request counts as making no change when the status, punch-in, and punch-out values remain the same after timestamp normalisation. An explicitly supplied `null` for `punched_at` is rejected because the field requires an integer.

**Department analytics:** I start the department summary from employees rather than attendance logs. This ensures employees without attendance records are still included in department headcounts.

**Database updates:** I use a MongoDB update pipeline to preserve correction history safely, including older records with a null history field. I also use indexes and database-side aggregation to avoid loading large datasets into Python.

**Time handling:** I use timezone-aware timestamps and apply attendance rules in IST. I chose synchronous PyMongo because it keeps the implementation straightforward, while database constraints protect against duplicate concurrent operations.