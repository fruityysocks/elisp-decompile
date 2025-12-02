 (defun max-list (list)
  (if (null list)
      0
      (if (null (cdr list))
          (car list)
          (let ((max (max-list (cdr list))))
            (if (> (car list) max)
                (car list)
                max)))))