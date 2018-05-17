/* Code provided by S. Cotton, U.Penn
 *
 *    N O T E !!!!  (Rada 06/11/01)
 * For this code to work, it is VERY important that all input files
 * (i.e. key, answer and sensemap) are SORTED 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#define TRUE 1
#define FALSE 0
#define NOT_FOUND -1

#define COMMENT_DELIMITER "!!"
#define WEIGHT_DELIMITER '/'

#define MINIMAL_SCORING_FLAG "-m"
#define VERBOSITY_FLAG "-v"
#define GRANULARITY_FLAG "-g"

#define COARSE_GRAIN_FLAG "coarse"
#define MIXED_GRAIN_FLAG "mixed"

#define UNASSIGNABLE_SENSE_TAG "U"
#define TYPO_SENSE_TAG "TYPO"
#define PROPER_SENSE_TAG "P"

typedef char boolean;

typedef struct
{FILE * pointer;
 char * name;
 char * line;
 long int length_of_line;} 
i_o_block;

typedef struct
{char ** index;
 char * buffer;
 long int last_entry;
 long int initial_bound [('z' - 'a') + 1] [('z' - 'a') + 1] [2];}
string_table;

void newline (void)

{printf ("\n");
 return;}

void fnewline (FILE * out)

{if (out != NULL) fprintf (out, "\n");
 return;}

boolean strequal (char * first_string, char * second_string)

{int i;
 
 if (first_string == NULL || second_string == NULL) return FALSE;

 for (i = 0; first_string [i] != '\0' && second_string [i] != '\0'; i++)
   if (first_string [i] != second_string [i]) break;

 return (first_string [i] == second_string [i]);}

char * make_buffer_of_size (long int length)

{if (length <= 0) length = 1;
 return (char *) calloc (length, sizeof (char));}

void reclaim (void * space_to_reclaim)

{if (space_to_reclaim != NULL) free (space_to_reclaim);

 return;}

void copystr (char ** target_string, char * source_string)

{int i;

 if (target_string == NULL || source_string == NULL) return;

 reclaim (*target_string);

 i = 0;
 while (source_string [i] != '\0') i++;

 *target_string = (char *) calloc (i + 1, sizeof (char));

 while (i >= 0)
  {*(*target_string + i) = source_string [i];
   i--;}

 return;}

void addstr (char ** target_string, char * source_string)

{char * new_string;
 int i;
 int j;

 if (target_string == NULL || source_string == NULL) return;

 i = 0;
 if (*target_string != NULL)
   while (*(*target_string + i) != '\0') 
     i++;
 
 j = 0;
 while (source_string [j] != '\0') j++;

 new_string = (char *) calloc (i + j + 1, sizeof (char));

 i = 0;

 if (*target_string != NULL)
   while (*(*target_string + i) != '\0')
    {new_string [i] = *(*target_string + i);
     i++;}

 for (j = 0; source_string [j] != '\0'; j++)
  {new_string [i] = source_string [j];
   i++;}

 new_string [i] = '\0';

 reclaim (*target_string);
 *target_string = new_string;

 return;}

boolean has_prefix (char * line, char * prefix)

{int i;

 if (line == NULL || prefix == NULL) return FALSE;

 for (i = 0; prefix [i] != '\0'; i++)
   if (line [i] != prefix [i]) break;

 return (prefix [i] == '\0');}

void strip_carriage_return (char * string_to_be_stripped)

{int length = 0;

 if (string_to_be_stripped == NULL) return;

 while (string_to_be_stripped [length] != '\0') length++;
 length--;
 if (string_to_be_stripped [length] == '\n') string_to_be_stripped [length] = '\0';

 return;}

boolean unreadable_file (char * file_name)

{FILE * file_to_be_tested;

 if (file_name == NULL) return TRUE;

 if ((file_to_be_tested = fopen (file_name, "r")) != NULL)
  {fclose (file_to_be_tested);
   return FALSE;}
 else
  {fclose (file_to_be_tested);
   return TRUE;}}

long int length_of_file (char * file_name)

{FILE * file_to_be_measured;
 long int counter = 0;

 if (file_name != NULL)

  {if ((file_to_be_measured = fopen (file_name, "r")) != NULL)
    {while (fgetc (file_to_be_measured) != EOF) counter++;
     counter++;}   

   fclose (file_to_be_measured);}

 return counter + 1;}

long int length_of_longest_line_in (char * file_name)

{FILE * file_to_be_measured;
 char file_character;
 long int max = 0;
 long int i;

 if (file_name != NULL)

  {if ((file_to_be_measured = fopen (file_name, "r")) != NULL)
    {i = 1;
     while ((file_character = fgetc (file_to_be_measured)) != EOF)
      {if (file_character == '\n')
        {if (i > max) max = i;
         i = 1;}
       else
         i++;}
     if (i > max) max = i;}

   fclose (file_to_be_measured);}

 return max + 1;}

boolean open_file (i_o_block * file_to_open, char * mode)

{if (file_to_open != NULL) 
  {file_to_open -> length_of_line = length_of_longest_line_in (file_to_open -> name);
   file_to_open -> line = make_buffer_of_size (file_to_open -> length_of_line);
   if (mode != NULL)
     file_to_open -> pointer = fopen (file_to_open -> name, mode);
   else
     file_to_open -> pointer = NULL;
   return (file_to_open -> pointer != NULL);}

 return FALSE;}
 
void close_file (i_o_block * file_to_close)
 
{if (file_to_close != NULL)
  {fclose (file_to_close -> pointer);
   file_to_close -> pointer = NULL;
   file_to_close -> length_of_line = 0;
   reclaim (file_to_close -> line);
   file_to_close -> line = NULL;
   reclaim (file_to_close -> name);
   file_to_close -> name = NULL;}
 
 return;} 

boolean read_line (i_o_block * file_to_read)

{if (file_to_read == NULL || file_to_read -> line == NULL || file_to_read -> pointer == NULL ||
     fgets (file_to_read -> line, file_to_read -> length_of_line, file_to_read -> pointer) == NULL)
   return FALSE;

 strip_carriage_return (file_to_read -> line);
 return TRUE;}

string_table * make_string_table (void)

{string_table * new;
 int i;
 int j;

 new = (string_table *) malloc (sizeof (string_table));

 new -> index = NULL;
 new -> buffer = NULL;
 new -> last_entry = 0;

 for (i = 0; i < ('z' - 'a') + 1; i++)
   for (j = 0; j < ('z' - 'a') + 1; j++)
    {new -> initial_bound [i] [j] [0] = 0;
     new -> initial_bound [i] [j] [1] = 0;}

 return new;}
 
void reclaim_string_table (string_table * table_to_reclaim)

{if (table_to_reclaim != NULL)

  {reclaim (table_to_reclaim -> index);
   reclaim (table_to_reclaim -> buffer);
   reclaim (table_to_reclaim);}

 return;}

void build_initial_bounds_of_table (string_table * table_to_build)

{char * table_entry;
 int current_first_letter = NOT_FOUND;
 int current_second_letter = NOT_FOUND;
 int i;

 if (table_to_build != NULL)
  {for (i = 0; i < table_to_build -> last_entry; i++)
    {table_entry = table_to_build -> index [i];
     if (table_entry [0] < 'a' || table_entry [0] > 'z' ||
         table_entry [1] < 'a' || table_entry [1] > 'z')
      {if (current_first_letter != NOT_FOUND)
        {table_to_build -> initial_bound [current_first_letter] [current_second_letter] [1] = i;
         current_first_letter = NOT_FOUND;}}
     else
      {if (table_entry [0] - 'a' != current_first_letter ||
           table_entry [1] - 'a' != current_second_letter)
        {if (current_first_letter != NOT_FOUND)
           table_to_build -> initial_bound [current_first_letter] [current_second_letter] [1] = i;
         current_first_letter = table_entry [0] - 'a';
         current_second_letter = table_entry [1] - 'a';
         table_to_build -> initial_bound [current_first_letter] [current_second_letter] [0] = i;}}}
        
   if (current_first_letter != NOT_FOUND)
     table_to_build -> initial_bound [current_first_letter] [current_second_letter] [1] = i;}

 return;}

long int search_string_table (char * word, string_table * table_to_search)

{long int high_bound;
 long int low_bound;
 long int new_middle;
 long int i;
 int j;

if (word == NULL || table_to_search == NULL || table_to_search -> index == NULL)
   return NOT_FOUND;

 if (word [0] >= 'a' && word [0] <= 'z' &&
     word [1] >= 'a' && word [1] <= 'z')
  {high_bound = table_to_search -> initial_bound [word [0] - 'a'] [word [1] - 'a'] [1];
   if (high_bound == 0) return NOT_FOUND;
   low_bound = table_to_search -> initial_bound [word [0] - 'a'] [word [1] - 'a'] [0];}
 else
  {high_bound = table_to_search -> last_entry;
   low_bound = 0;}

 new_middle = (high_bound + low_bound) / 2;
  
 do
  {i = new_middle;
  
  for (j = 0; word [j] != '\0'; j++)
     {
       if (word [j] < table_to_search -> index [i] [j])
      {high_bound = i;
       break;}
     if (word [j] > table_to_search -> index [i] [j])
      {low_bound = i;
       break;}}
   if (word [j] == '\0')
    {if (table_to_search -> index [i] [j] == '\0' ||
         table_to_search -> index [i] [j] == ' ')
       return i;
     else
       high_bound = i;}
   new_middle = (high_bound + low_bound) / 2;}
 while (i != new_middle);

 return NOT_FOUND;}


string_table * unsorted_string_table_for (char * file_name)

{string_table * new;
 FILE * words;
 long int beginning_of_word;
 long int characters_seen;
 char file_character;

 new = make_string_table ();

 characters_seen = 0;

 if (file_name != NULL)
  {if ((words = fopen (file_name, "r")) != NULL)
     while ((file_character = fgetc (words)) != EOF)
      {if (file_character == '\n') 
        (new -> last_entry)++;
       characters_seen++;}
   fclose (words);}

 new -> buffer = make_buffer_of_size (characters_seen + 1);
 new -> buffer [0] = '\0';

 new -> index = (char **) calloc ((new -> last_entry) + 1, sizeof (char *));
 new -> index [0] = new -> buffer;

 new -> last_entry = 0;
 characters_seen = 0;
 beginning_of_word = 0;

 if (file_name != NULL)
  {if ((words = fopen (file_name, "r")) != NULL)
    while ((new -> buffer [characters_seen] = fgetc (words)) != EOF)
      {if (new -> buffer [characters_seen] == '\n')
        {new -> buffer [characters_seen] = '\0';
         new -> index [new -> last_entry] = &(new -> buffer [beginning_of_word]);
         beginning_of_word = characters_seen + 1;
        (new -> last_entry)++;}
       characters_seen++;}
   fclose (words);}

 return new;}

string_table * string_table_for (char * file_name)

{string_table * new;

 new = unsorted_string_table_for (file_name);

 build_initial_bounds_of_table (new);

 return new;}

char ** tokenize (char * line)

{char ** tokens;
 int last_token = 0;
 int i;

 if (line != NULL) 
   tokens = (char **) calloc (strlen (line) + 1, sizeof (char *));
 else
   tokens = (char **) calloc (1, sizeof (char *));

 tokens [last_token] = NULL;

 if (line != NULL)
   for (i = 0; line [i] != '\0'; i++)
    {if (line [i] == ' ' || line [i] == '\n' || line [i] == '\t')
      {line [i] = '\0';    
       if (tokens [last_token] != NULL) 
        {last_token++;
         tokens [last_token] = NULL;}}

     else
     if (tokens [last_token] == NULL)
       tokens [last_token] = &(line [i]);}

 if (tokens [last_token] != NULL) 
  {last_token++;
   tokens [last_token] = NULL;}

 return tokens;}

char * coarse_sense_tag_for (char * tag, string_table * sense_table)

{char * coarse_tag;
 long int sense_index;
 int i;
 
 coarse_tag = tag;

 sense_index = search_string_table (tag, sense_table);
 if (sense_index != NOT_FOUND)
  {coarse_tag = sense_table -> index [sense_index];
   for (i = 0; sense_table -> index [sense_index] [i] != '\0'; i++)
     if (sense_table -> index [sense_index] [i] == ' ')
       coarse_tag = &(sense_table -> index [sense_index] [i + 1]);}

 return coarse_tag;}

float subsumption_probability_of_tags (char * subsumer, char * tag, string_table * sense_table)

{float probability;
 long int sense_index;
 int i;
 int j;
 
 sense_index = search_string_table (tag, sense_table);

 if (sense_index != NOT_FOUND && subsumer != NULL)
  {probability = 1.0;
   i = 0;
   for (j = 0; sense_table -> index [sense_index] [j] != '\0'; j++)
     if (j == 0 || sense_table -> index [sense_index] [j - 1] == ' ')
      {i++;
       if (i % 2 == 0)
         probability *= atof (&(sense_table -> index [sense_index] [j]));
       else
       if (has_prefix (&(sense_table -> index [sense_index] [j]), subsumer) &&
           (sense_table -> index [sense_index] [j + strlen (subsumer)] == ' ' ||
            sense_table -> index [sense_index] [j + strlen (subsumer)] == '\0'))
         return 1.0 / probability;}}

 return 0.0;}

int main (int argc, char * argv [])

{

  boolean invalid_argument_seen = FALSE;
  string_table * answer;
  string_table * key;
  string_table * sense_table;
  int sense_table_arg = 0;
  char *** key_tag;
  char ** answer_tag;
  float * answer_weight;
  long int answer_index;
  long int key_index;
  long int sense_index;
  char * sense_granularity = NULL;
  char * coarse_sense_tag;
  boolean minimal_scoring_desired = FALSE;
  boolean line_by_line_trace_desired = FALSE;
  float total_weight;
  float instances_attempted;
  float instances;
  float score;
  float subscore;
  float global_score;
  int i;
  int j;
  int k;

 if (argc < 3)
  {fnewline (stderr);   
   fprintf (stderr, "This scorer requires two command-line arguments in the");
   fnewline (stderr);   
   fprintf (stderr, "following order:");
   fnewline (stderr);   
   fnewline (stderr);
   fprintf (stderr, "   ANSWER FILE NAME (name of a file containing formatted answers)");
   fnewline (stderr);
   fnewline (stderr);   
   fprintf (stderr, "   KEY FILE NAME (name of an answer-key file)");
   fnewline (stderr);
   fnewline (stderr);   
   fprintf (stderr, "Optionally, the following may be appended:");
   fnewline (stderr);
   fnewline (stderr);
   fprintf (stderr, "   SENSE FILE NAME (name of a file containing sense-map information)");
   fnewline (stderr);
   fprintf (stderr, "     - without this file, only fine-grained scoring is available,\n");
   fprintf (stderr, "       and illformed sense tags will lower precision (rather than recall)");
   fnewline (stderr);   
   fnewline (stderr);
   fprintf (stderr, "   %s", GRANULARITY_FLAG);
   fprintf (stderr, " (specifies granularity: ");
   fprintf (stderr, "\"%s\" or \"%s\"; \"fine\" is default)", COARSE_GRAIN_FLAG, MIXED_GRAIN_FLAG);
   fnewline (stderr);   
   fnewline (stderr);
   fprintf (stderr, "   %s", MINIMAL_SCORING_FLAG);
   fprintf (stderr, " (causes exclusion of instances tagged with multiple tags in key)");
   fnewline (stderr);   
   fnewline (stderr);
   fprintf (stderr, "   %s", VERBOSITY_FLAG);
   fprintf (stderr, " (causes line-by-line scoring calculations to be printed)");
   fnewline (stderr);   
   fnewline (stderr);
   invalid_argument_seen = TRUE;}

 for (i = 1; i <= 2 && i < argc; i++)
   if (unreadable_file (argv [i])) {
     fnewline (stderr);
     fprintf (stderr, "Unable to read \"%s\".", argv [i]);
     fnewline (stderr);
     fnewline (stderr);
     invalid_argument_seen = TRUE;
   }

 for (i = 3; i < argc; i++) {
   if (argv[i][0] != '-') {
     if (unreadable_file (argv [i])) {
       fnewline (stderr);
       fprintf (stderr, "Unable to read \"%s\".", argv [i]);
       fnewline (stderr);
       fnewline (stderr);
       invalid_argument_seen = TRUE;
     } else {
       sense_table_arg = i;
     }
   }
   if (strequal (argv [i], MINIMAL_SCORING_FLAG)) minimal_scoring_desired = TRUE;
   if (strequal (argv [i], VERBOSITY_FLAG)) line_by_line_trace_desired = TRUE;
   if (strequal (argv [i], GRANULARITY_FLAG) && i + 1 < argc &&
       (strequal (argv [i + 1], COARSE_GRAIN_FLAG) || strequal (argv [i + 1], MIXED_GRAIN_FLAG)))
     sense_granularity = argv [++i];
 }

 if (sense_table_arg == 0) {
   /* force fine-grained scoring */
   if (sense_granularity) {
     fprintf(stderr,"Fine-grained scoring will be used since no sense map was given\n");
     sense_granularity = NULL;
   }
 }

 if (!(invalid_argument_seen)) {

   global_score = 0.0;
   instances = 0.0;
   instances_attempted = 0.0;
   
   if (sense_table_arg > 0) {
     sense_table = string_table_for (argv [sense_table_arg]);
   } else {
     sense_table = NULL;
   }

   key = string_table_for (argv [2]);
   key_tag = (char ***) calloc (key -> last_entry + 1, sizeof (char ***));

   for (key_index = 0; key_index < key -> last_entry; key_index++) {

     /* Re-format the key's first two fields, which specify the test instance to be answered */
     
     for (i = 0; key -> index [key_index] [i] != '\0'; i++)
       if (key -> index [key_index] [i] == ' ') {
	 key -> index [key_index] [i] = '_';
         while (key -> index [key_index] [i] != '\0' &&
		key -> index [key_index] [i] != ' ')
	   {i++;}
	 break;
       }
	 
     if (key -> index [key_index] [i] == ' ') {

       key_tag [key_index] = tokenize (&(key -> index [key_index] [i]));

       /* Convert correct answers to appropriate granularity, collapsing duplicates if they arise */

       if (strequal (sense_granularity, COARSE_GRAIN_FLAG))
	 for (i = 0; key_tag [key_index] [i] != NULL; i++) {
	   coarse_sense_tag = coarse_sense_tag_for (key_tag [key_index] [i], sense_table);
	   for (j = 0; j < i; j++)
	     if (strequal (coarse_sense_tag, key_tag [key_index] [j])) break;
	   if (j < i) {
	     for (j = i; key_tag [key_index] [j] != NULL; j++)
	       key_tag [key_index] [j] = key_tag [key_index] [j + 1];
	     i--;
	   } 
	   else
             if (!(strequal (coarse_sense_tag, key_tag [key_index] [i])))
               key_tag [key_index] [i] = coarse_sense_tag;
	 }
       
       /* If filter criteria for this instance aren't matched, mark the key to exclude it from scoring */
       
       if (minimal_scoring_desired && key_tag [key_index] [0] != NULL && key_tag [key_index] [1] != NULL)
	 key_tag [key_index] [0] = NULL;
       
       if (key_tag [key_index] [0] != NULL) instances += 1.0;

       /*break;*/
     }
   }

   answer = string_table_for (argv [1]);

   for (answer_index = 0; answer_index < answer -> last_entry; answer_index++)

     /* Strip off any comments in answers */

    {for (i = 0; answer -> index [answer_index] [i] != '\0'; i++)
       if (has_prefix (&(answer -> index [answer_index] [i]), COMMENT_DELIMITER))
        {answer -> index [answer_index] [i] = '\0';
         break;}
     
     /* Re-format the answer's first two fields, which specify the test instance which is being answered */

     for (i = 0; answer -> index [answer_index] [i] != '\0'; i++)
       if (answer -> index [answer_index] [i] == ' ')
        {answer -> index [answer_index] [i] = '_';
         while (answer -> index [answer_index] [i] != '\0')
          {if (answer -> index [answer_index] [i] == ' ') 
            {answer -> index [answer_index] [i] = '\0';
             i++;
             break;}
           i++;}
         break;}

     /* Parse the raw answer, separating out sense tags and their associated weights, if any */

     answer_tag = tokenize (&(answer -> index [answer_index] [i]));

     i = 0; 
     while (answer_tag [i] != NULL) i++;
     answer_weight = (float *) calloc (i + 1, sizeof (float));

     total_weight = 0.0;
     invalid_argument_seen = FALSE;
     for (i = 0; answer_tag [i] != NULL; i++)

      {for (j = 0; answer_tag [i] [j] != '\0'; j++)
         if (answer_tag [i] [j] == WEIGHT_DELIMITER) break;

       if (answer_tag [i] [j] != WEIGHT_DELIMITER)
         invalid_argument_seen = TRUE;
       else
        {answer_tag [i] [j] = '\0';
         answer_weight [i] = atof (&(answer_tag [i] [j + 1]));
         total_weight += answer_weight [i];}

       /* Discard any answer which contains an unknown sense tag, or a sense tag with a suffix */
       /* But don't check if we don't have a sense table */

       /*if (sense_table) {
	 sense_index = search_string_table (answer_tag [i], sense_table);
	 if (sense_index == NOT_FOUND &&
	     !(strequal (answer_tag [i], UNASSIGNABLE_SENSE_TAG)) &&
	     !(strequal (answer_tag [i], TYPO_SENSE_TAG)) &&
	     !(strequal (answer_tag [i], PROPER_SENSE_TAG))) {
	   answer_tag [0] = NULL;
	   break;
	 }} */
      }

     /* If not all sense tags have weights, set the weights for all of them to be equal */

     if (invalid_argument_seen)
      {for (i = 0; answer_tag [i] != NULL; i++) answer_weight [i] = 1.0;
       total_weight = 1.0 * i;}

     /* Convert answer tags to appropriate granularity, collapsing duplicates and merging their weights */

     if (strequal (sense_granularity, COARSE_GRAIN_FLAG))
       for (i = 0; answer_tag [i] != NULL; i++)
        {coarse_sense_tag = coarse_sense_tag_for (answer_tag [i], sense_table);
	for (j = 0; j < i; j++)
           if (strequal (coarse_sense_tag, answer_tag [j])) break;
         if (j < i)
          {answer_weight [j] += answer_weight [i];
           for (j = i; answer_tag [j] != NULL; j++)
            {answer_tag [j] = answer_tag [j + 1];
             answer_weight [j] = answer_weight [j + 1];}
           i--;} 
         else
         if (!(strequal (coarse_sense_tag, answer_tag [i])))
           answer_tag [i] = coarse_sense_tag;}

     /* If weights don't form a probability distribution on sense tags (plus implicit no-guess tag), normalize them */

     if (total_weight > 1.0) 
      {for (i = 0; answer_tag [i] != NULL; i++) answer_weight [i] /= total_weight;
       total_weight = 1.0;}

     /* Look up the correct answer for this test instance in the key */

     key_index = search_string_table (answer -> index [answer_index], key);
     
     /* If there is such a test instance, and it hasn't been scored already or excluded by filter criteria */

     if (key_index != NOT_FOUND && answer_tag [0] != NULL && key_tag [key_index] [0] != NULL)

      {instances_attempted += total_weight;

       score = 0.0;

       /* Check each answer sense tag against the correct sense tags in the key */

       for (i = 0; answer_tag [i] != NULL; i++)

        {subscore = 0.0;

         for (j = 0; key_tag [key_index] [j] != NULL; j++)

           /* If a tag in the key is equal to the answer tag, give full credit */

          {if (strequal (answer_tag [i], key_tag [key_index] [j]))
             subscore += 1.0;
           else
	     
	     /* If correct tag subsumes the answer, give full credit under mixed-grain scoring */

           if (strequal (sense_granularity, MIXED_GRAIN_FLAG) &&
               subsumption_probability_of_tags (key_tag [key_index] [j], answer_tag [i], sense_table) > 0.0)
             subscore += 1.0;
           else

           /* If answer subsumes correct tag, give partial credit under mixed-grain scoring */

           if (strequal (sense_granularity, MIXED_GRAIN_FLAG))
             subscore += subsumption_probability_of_tags (answer_tag [i], key_tag [key_index] [j], sense_table);}
  
	 
         if (subscore > 1.0) subscore = 1.0;

         score += subscore * answer_weight [i];}

       if (score > 1.0) score = 1.0;

       global_score += score;

       if (line_by_line_trace_desired)
        {printf ("score for \"%s\": %.3f", key -> index [key_index], score);
         newline ();
         printf (" key   =");
         for (i = 0; key_tag [key_index] [i] != NULL; i++) 
           printf (" %s", key_tag [key_index] [i]);
         newline ();
         printf (" guess =");
         for (i = 0; answer_tag [i] != NULL; i++) 
           printf (" %s%c%.3f", answer_tag [i], WEIGHT_DELIMITER, answer_weight [i]);
         newline ();
         newline ();}

       /* Reset the key for this test instance so a second answer to this instance won't be counted redundantly */

       key_tag [key_index] [0] = NULL;}
   
     else

      {fprintf (stderr, "Unable to score answer for \"%s\" (line %ld):", answer -> index [answer_index], answer_index);
       if (key_index == NOT_FOUND)
         fprintf (stderr, " Does not exist.");
       else
       if (answer_tag [0] == NULL)
         fprintf (stderr, " Invalid answer.");
       else
         fprintf (stderr, " Excluded or already scored.");
       fnewline (stderr);

       if (line_by_line_trace_desired)
        {printf ("Unable to score answer for \"%s\" (line %ld):", answer -> index [answer_index], answer_index);
         if (key_index == NOT_FOUND)
           printf (" Does not exist.");
         else
         if (answer_tag [0] == NULL)
           printf (" Invalid answer.");
         else
           printf (" Excluded or already scored.");
         newline ();
         newline ();}}

     reclaim (answer_tag);
     reclaim (answer_weight);}

   newline ();
   if (strequal (sense_granularity, COARSE_GRAIN_FLAG))
     printf ("Coarse-grained");
   else
   if (strequal (sense_granularity, MIXED_GRAIN_FLAG))
     printf ("Mixed-grained");
   else
     printf ("Fine-grained");
   if (minimal_scoring_desired) printf (" minimal");
   printf (" score for \"%s\" using key \"%s\":", argv [1], argv [2]);
   newline ();
   printf (" precision: %.3f", global_score / instances_attempted);
   printf (" (%.2f correct of %.2f attempted)", global_score, instances_attempted);
   newline ();
   printf (" recall: %.3f", global_score / instances);
   printf (" (%.2f correct of %.2f in total)", global_score, instances);
   newline ();
   printf (" attempted: %.2f %%", 100.0 * instances_attempted / instances);
   printf (" (%.2f attempted of %.2f in total)", instances_attempted, instances);
   newline ();
   newline ();

   reclaim_string_table (answer);

   for (key_index = 0; key_index < key -> last_entry; key_index++) reclaim (key_tag [key_index]);
   reclaim (key_tag);
   reclaim_string_table (key);

   reclaim_string_table (sense_table);
 }
}




